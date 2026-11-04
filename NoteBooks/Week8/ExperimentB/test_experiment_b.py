import os
import sys
import glob
import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../NoteBooks/Week6")))
from dataset import PediatricECGMultiWindowDataset
from models import MultiScaleResNet1D

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def patient_level_split(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
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
    for pid, patient_to_records_items in patient_to_records.items():
        if pid in train_patients:
            train_records.extend(patient_to_records_items)
        elif pid in val_patients:
            val_records.extend(patient_to_records_items)
        else:
            test_records.extend(patient_to_records_items)
            
    return train_records, val_records, test_records

def main():
    print(f"\n{'='*50}\nExperiment B: Final Test Evaluation (Multi-Scale)\n{'='*50}")
    set_seed(42)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    checkpoint_path = os.path.join(current_dir, "checkpoints", "exp_b_multiscale_best.pth")
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure training is complete before running testing.")
        return
        
    _, _, test_recs = patient_level_split(data_dir)
    print(f"Test Records: {len(test_recs)}")
    
    test_dataset = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=test_recs, num_windows=5)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleResNet1D(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    print("Running 5-window inference (MAX aggregation)...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            # inputs shape: (1, 5, 12, 5000)
            inputs = inputs.squeeze(0).to(device) # (5, 12, 5000)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1] # shape (5,)
            
            # MAX Aggregation
            max_prob = torch.max(probs).item()
            pred = 1 if max_prob >= 0.5 else 0
            
            all_labels.append(labels.item())
            all_probs.append(max_prob)
            all_preds.append(pred)
            
    auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    spec = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n" + "="*50)
    print("Experiment B - Test Set Performance")
    print("="*50)
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    
    with open(os.path.join(results_dir, "experiment_b_test_results.txt"), "w") as f:
        f.write(f"Experiment B Test Performance\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Sensitivity: {rec:.4f}\n")
        f.write(f"Specificity: {spec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        
    print(f"\nResults saved to {results_dir}/experiment_b_test_results.txt")

if __name__ == "__main__":
    main()
