import os
import glob
import wfdb
import numpy as np
from tqdm import tqdm

def patient_level_split(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    import random
    records = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
    records = [r[:-4] for r in records]
    
    patient_to_records = {}
    for r in records:
        pid = os.path.basename(os.path.dirname(r + ".hea"))
        if pid not in patient_to_records:
            patient_to_records[pid] = []
        patient_to_records[pid].append(r)
        
    unique_patients = list(patient_to_records.keys())
    unique_patients.sort()
    
    random.seed(seed)
    random.shuffle(unique_patients)
    
    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train+n_val])
    test_patients = set(unique_patients[n_train+n_val:])
    
    train_records, val_records, test_records = [], [], []
    for pid, recs in patient_to_records.items():
        if pid in train_patients:
            train_records.extend(recs)
        elif pid in val_patients:
            val_records.extend(recs)
        else:
            test_records.extend(recs)
            
    return train_patients, val_patients, test_patients, train_records, val_records, test_records

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    print("="*60)
    print("1. DATA SPLIT VERIFICATION")
    print("="*60)
    
    train_pat, val_pat, test_pat, train_rec, val_rec, test_rec = patient_level_split(data_dir)
    
    train_val_overlap = train_pat.intersection(val_pat)
    train_test_overlap = train_pat.intersection(test_pat)
    val_test_overlap = val_pat.intersection(test_pat)
    
    print(f"Number of Train patients:      {len(train_pat)}")
    print(f"Number of Validation patients: {len(val_pat)}")
    print(f"Number of Test patients:       {len(test_pat)}")
    print(f"Total Unique Patients:         {len(train_pat) + len(val_pat) + len(test_pat)}\n")
    
    print(f"Number of Train records:       {len(train_rec)}")
    print(f"Number of Validation records:  {len(val_rec)}")
    print(f"Number of Test records:        {len(test_rec)}")
    print(f"Total Records:                 {len(train_rec) + len(val_rec) + len(test_rec)}\n")
    
    print("Overlap Check:")
    print(f"Train ∩ Validation patients: {len(train_val_overlap)} -> {'PASS' if len(train_val_overlap) == 0 else 'FAIL'}")
    print(f"Train ∩ Test patients:       {len(train_test_overlap)} -> {'PASS' if len(train_test_overlap) == 0 else 'FAIL'}")
    print(f"Validation ∩ Test patients:  {len(val_test_overlap)} -> {'PASS' if len(val_test_overlap) == 0 else 'FAIL'}")
    
    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        print("ERROR: Patient leakage detected!")
        return
        
    print("\n" + "="*60)
    print("2. DURATION STATISTICS (Whole Dataset)")
    print("="*60)
    
    all_records = train_rec + val_rec + test_rec
    
    shorter, exact, longer = 0, 0, 0
    total = len(all_records)
    
    print("Analyzing lengths of all ECG recordings...")
    for rec in tqdm(all_records):
        header = wfdb.rdheader(rec)
        sig_len = header.sig_len
        
        if sig_len < 5000:
            shorter += 1
        elif sig_len == 5000:
            exact += 1
        else:
            longer += 1
            
    print("\nDuration Analysis (relative to 5000 samples / 10s):")
    print(f"Shorter than 10s: {shorter} ({shorter/total*100:.2f}%)")
    print(f"Exactly 10s:      {exact} ({exact/total*100:.2f}%)")
    print(f"Longer than 10s:  {longer} ({longer/total*100:.2f}%)")
    print(f"Total evaluated:  {total}")

if __name__ == "__main__":
    main()
