import torch
from torch.utils.data import Dataset
import wfdb
import glob
import os
import numpy as np
import scipy.signal as signal

class PediatricECGDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5000, apply_filter=False):
        self.records = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
        self.records = [r[:-4] for r in self.records]
        self.sequence_length = sequence_length
        self.apply_filter = apply_filter
        
        print(f"Scanning dataset. Preprocessing filter enabled: {self.apply_filter}")
        self.labels = []
        for r in self.records:
            _, fields = wfdb.rdsamp(r, sampto=1)
            comments = fields.get('comments', [])
            diagnosis = comments[-1].lower() if comments else ""
            self.labels.append(0 if "normal ecg" in diagnosis else 1)
            
    def _apply_zero_phase_filter(self, ecg_signal, fs=500):
        # Phase 2: Rationale - Butterworth Zero-Phase Filtering (filtfilt)
        # Low cutoff: 0.5 Hz (Removes baseline wander / respiration)
        # High cutoff: 40 Hz (Removes high-frequency muscle noise / powerline interference)
        nyq = 0.5 * fs
        low = 0.5 / nyq
        high = 40.0 / nyq
        b, a = signal.butter(N=3, Wn=[low, high], btype='bandpass')
        
        filtered_signal = np.zeros_like(ecg_signal)
        # Apply filter to each lead independently
        for i in range(ecg_signal.shape[0]):
            # filtfilt applies the filter forward and backward to ensure ZERO phase distortion
            filtered_signal[i, :] = signal.filtfilt(b, a, ecg_signal[i, :])
            
        return filtered_signal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        signals, _ = wfdb.rdsamp(record_path)
        signals = np.transpose(signals)
        
        # Apply Zero-Phase Filtering if configured for this experiment
        if self.apply_filter:
            signals = self._apply_zero_phase_filter(signals)
        
        # Standardize Channels to 12
        target_channels = 12
        if signals.shape[0] > target_channels:
            signals = signals[:target_channels, :]
        elif signals.shape[0] < target_channels:
            pad_channels = target_channels - signals.shape[0]
            signals = np.pad(signals, ((0, pad_channels), (0, 0)), mode='constant')
            
        # Standardize Sequence Length to exactly 5000
        if signals.shape[1] > self.sequence_length:
            signals = signals[:, :self.sequence_length]
        elif signals.shape[1] < self.sequence_length:
            pad_width = self.sequence_length - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
            
        # Normalization (Z-score scaling per lead) to stabilize deep networks
        means = np.mean(signals, axis=1, keepdims=True)
        stds = np.std(signals, axis=1, keepdims=True)
        # Avoid division by zero for padded empty leads
        stds[stds == 0] = 1 
        signals = (signals - means) / stds
            
        return torch.tensor(signals, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
