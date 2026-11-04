import torch
from torch.utils.data import Dataset
import wfdb
import glob
import os
import numpy as np
import scipy.signal as signal

class PediatricECGDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5000, apply_filter=False, record_paths=None, random_crop=False):
        """
        If record_paths is provided, it uses those explicitly (useful for train/val/test patient splits).
        Otherwise, it globs the entire directory.
        """
        if record_paths is not None:
            self.records = record_paths
        else:
            self.records = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
            self.records = [r[:-4] for r in self.records]
            
        self.sequence_length = sequence_length
        self.apply_filter = apply_filter
        self.random_crop = random_crop
        
        # Extract Patient IDs from filenames (e.g., .../P00001/P00001_E01 -> P00001)
        self.patient_ids = [os.path.basename(os.path.dirname(r + ".hea")) for r in self.records]
        
        print(f"Scanning dataset ({len(self.records)} records). Preprocessing filter enabled: {self.apply_filter}")
        self.labels = []
        for r in self.records:
            _, fields = wfdb.rdsamp(r, sampto=1)
            comments = fields.get('comments', [])
            diagnosis = comments[-1].lower() if comments else ""
            self.labels.append(0 if "normal ecg" in diagnosis else 1)
            
    def _apply_zero_phase_filter(self, ecg_signal, fs=500):
        nyq = 0.5 * fs
        low = 0.5 / nyq
        high = 40.0 / nyq
        b, a = signal.butter(N=3, Wn=[low, high], btype='bandpass')
        
        filtered_signal = np.zeros_like(ecg_signal)
        for i in range(ecg_signal.shape[0]):
            filtered_signal[i, :] = signal.filtfilt(b, a, ecg_signal[i, :])
            
        return filtered_signal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        signals, _ = wfdb.rdsamp(record_path)
        signals = np.transpose(signals)
        
        if self.apply_filter:
            signals = self._apply_zero_phase_filter(signals)
        
        # Standardize Channels to 12
        target_channels = 12
        if signals.shape[0] > target_channels:
            signals = signals[:target_channels, :]
        elif signals.shape[0] < target_channels:
            pad_channels = target_channels - signals.shape[0]
            signals = np.pad(signals, ((0, pad_channels), (0, 0)), mode='constant')
            
        # Standardize Sequence Length to exactly 5000 (Phase 1/2 Baseline)
        if signals.shape[1] > self.sequence_length:
            if self.random_crop:
                max_start = signals.shape[1] - self.sequence_length
                import random
                start_idx = random.randint(0, max_start)
                signals = signals[:, start_idx:start_idx + self.sequence_length]
            else:
                signals = signals[:, :self.sequence_length]
        elif signals.shape[1] < self.sequence_length:
            pad_width = self.sequence_length - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
            
        # Normalization (Z-score scaling per lead)
        means = np.mean(signals, axis=1, keepdims=True)
        stds = np.std(signals, axis=1, keepdims=True)
        stds[stds == 0] = 1 
        signals = (signals - means) / stds
            
        return torch.tensor(signals, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class PediatricECGMultiWindowDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5000, apply_filter=False, record_paths=None, num_windows=3):
        if record_paths is not None:
            self.records = record_paths
        else:
            self.records = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
            self.records = [r[:-4] for r in self.records]
            
        self.sequence_length = sequence_length
        self.apply_filter = apply_filter
        self.num_windows = num_windows
        
        self.patient_ids = [os.path.basename(os.path.dirname(r + ".hea")) for r in self.records]
        
        print(f"Scanning Multi-Window dataset ({len(self.records)} records). Max windows: {self.num_windows}")
        self.labels = []
        for r in self.records:
            _, fields = wfdb.rdsamp(r, sampto=1)
            comments = fields.get('comments', [])
            diagnosis = comments[-1].lower() if comments else ""
            self.labels.append(0 if "normal ecg" in diagnosis else 1)
            
    def _apply_zero_phase_filter(self, ecg_signal, fs=500):
        nyq = 0.5 * fs
        low = 0.5 / nyq
        high = 40.0 / nyq
        b, a = signal.butter(N=3, Wn=[low, high], btype='bandpass')
        filtered_signal = np.zeros_like(ecg_signal)
        for i in range(ecg_signal.shape[0]):
            filtered_signal[i, :] = signal.filtfilt(b, a, ecg_signal[i, :])
        return filtered_signal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        signals, _ = wfdb.rdsamp(record_path)
        signals = np.transpose(signals)
        
        if self.apply_filter:
            signals = self._apply_zero_phase_filter(signals)
        
        target_channels = 12
        if signals.shape[0] > target_channels:
            signals = signals[:target_channels, :]
        elif signals.shape[0] < target_channels:
            pad_channels = target_channels - signals.shape[0]
            signals = np.pad(signals, ((0, pad_channels), (0, 0)), mode='constant')
            
        sig_len = signals.shape[1]
        windows = []
        
        if sig_len <= self.sequence_length:
            # Too short or exactly 10s -> 1 window only
            pad_width = self.sequence_length - sig_len
            w = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
            windows.append(w)
        else:
            # Longer than 10s -> multiple distinct windows
            # Calculate evenly spaced starts
            max_start = sig_len - self.sequence_length
            
            # If the max start allows for fewer non-overlapping/distinct windows than requested, 
            # we just take evenly spaced valid starts up to num_windows
            # The prompt says: "use the available distinct positions" "avoid duplicate meaningless windows"
            # For 3 windows, start indices could be 0, max_start//2, max_start
            if self.num_windows == 1:
                starts = [0]
            elif self.num_windows == 2:
                starts = [0, max_start]
            else:
                starts = np.linspace(0, max_start, min(self.num_windows, max_start + 1), dtype=int)
                starts = np.unique(starts) # Ensure strict distinctness
            
            for start_idx in starts:
                w = signals[:, start_idx:start_idx + self.sequence_length]
                windows.append(w)
                
        # Normalize each window independently (Z-score)
        norm_windows = []
        for w in windows:
            means = np.mean(w, axis=1, keepdims=True)
            stds = np.std(w, axis=1, keepdims=True)
            stds[stds == 0] = 1 
            w = (w - means) / stds
            norm_windows.append(w)
            
        # norm_windows is a list of [12, 5000] numpy arrays
        windows_tensor = torch.tensor(np.stack(norm_windows), dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return windows_tensor, label_tensor

def multi_window_collate_fn(batch):
    # batch is a list of tuples: (windows_tensor, label_tensor)
    # windows_tensor shape: [k, 12, 5000] where k can vary per record
    
    windows_list = []
    labels_list = []
    counts = []
    
    for windows, label in batch:
        windows_list.append(windows)
        labels_list.append(label)
        counts.append(windows.shape[0])
        
    # Concatenate all windows across the batch into a single tensor [sum(k), 12, 5000]
    batch_windows = torch.cat(windows_list, dim=0)
    batch_labels = torch.stack(labels_list)
    
    return batch_windows, batch_labels, counts
