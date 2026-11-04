import wfdb
import os
import glob
from dataset import PediatricECGMultiWindowDataset

def test_extraction():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    # We'll just grab a few records from the directory
    records = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
    records = [r[:-4] for r in records]
    
    # Try to find a short one (<5000), a 10s one (==5000), and a long one (>5000)
    # Based on earlier stats, almost all are > 5000. Let's just pick 3 records.
    dataset = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=records[:5], num_windows=3)
    
    for i in range(3):
        rec_path = dataset.records[i]
        header = wfdb.rdheader(rec_path)
        sig_len = header.sig_len
        
        windows, label = dataset[i]
        
        print("-" * 50)
        print(f"Record: {os.path.basename(rec_path)}")
        print(f"Original samples: {sig_len}")
        print(f"Window length: 5000")
        print(f"Number of windows: {windows.shape[0]}")
        
        # Infer start positions based on how dataset.py calculates them
        if sig_len <= 5000:
            print("Window start positions: [0]")
            print("Window end positions: [5000 (padded)]")
        else:
            max_start = sig_len - 5000
            import numpy as np
            starts = np.linspace(0, max_start, min(3, max_start + 1), dtype=int)
            starts = np.unique(starts)
            ends = [s + 5000 for s in starts]
            print(f"Window start positions: {starts.tolist()}")
            print(f"Window end positions: {ends}")
            
        print(f"Label: {label.item()}")
        print(f"Tensor Shape: {windows.shape}")
        
if __name__ == "__main__":
    test_extraction()
