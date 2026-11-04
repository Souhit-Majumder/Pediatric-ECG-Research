import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve
from tqdm import tqdm
from dataset import PediatricECGMultiWindowDataset, multi_window_collate_fn
from models import ResNet1D

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
    for pid, pr in patient_to_records.items():
        if pid in train_patients:
            train_records.extend(pr)
        elif pid in val_patients:
            val_records.extend(pr)
        else:
            test_records.extend(pr)
            
    return train_records, val_records, test_records

def evaluate_multi_window(model, dataloader, device, aggregation='mean'):
    model.eval()
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    total_windows = 0
    windows_counts = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_windows, batch_labels, counts in tqdm(dataloader, desc=f"Test Inference ({aggregation})"):
            batch_windows = batch_windows.to(device)
            
            outputs = model(batch_windows)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            idx = 0
            for label, c in zip(batch_labels, counts):
                record_probs = probs[idx:idx+c]
                
                if aggregation == 'mean':
                    agg_prob = record_probs.mean().item()
                elif aggregation == 'max':
                    agg_prob = record_probs.max().item()
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
                    
                agg_pred = 1 if agg_prob >= 0.5 else 0
                
                all_probs.append(agg_prob)
                all_preds.append(agg_pred)
                all_labels.append(label.item())
                
                windows_counts.append(c)
                total_windows += c
                idx += c
                
    end_time = time.time()
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'pr_auc': average_precision_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': sensitivity,  # Recall == Sensitivity
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'cm': cm,
        'all_probs': all_probs,
        'all_labels': all_labels,
        'runtime': end_time - start_time,
        'total_windows': total_windows,
        'windows_counts': windows_counts
    }
    
    return metrics

def plot_curves(labels, probs, plot_dir, prefix):
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    
    # ROC Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(labels, probs):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} - ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_roc_curve.png"))
    plt.close()
    
    # PR Plot
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {average_precision_score(labels, probs):.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_pr_curve.png"))
    plt.close()

def plot_confusion_matrix(cm, plot_dir, prefix):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{prefix} - Confusion Matrix')
    plt.colorbar()
    classes = ['Normal (0)', 'Abnormal (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()

def main():
    print(f"\n{'='*50}\nStarting Phase 3: FINAL BLIND TEST EVALUATION\n{'='*50}")
    set_seed(42)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    phase3_dir = os.path.join(current_dir, "Phase3")
    results_dir = os.path.join(phase3_dir, "results")
    plot_dir = os.path.join(phase3_dir, "plots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Splits
    _, _, test_recs = patient_level_split(data_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # 2. Model Loading
    model = ResNet1D(num_classes=2).to(device)
    phase2_model_path = os.path.join(current_dir, "Phase2", "checkpoints", "phase2_resnet_best.pth")
    if not os.path.exists(phase2_model_path):
        print(f"ERROR: Phase 2 model checkpoint not found at {phase2_model_path}!")
        return
        
    model.load_state_dict(torch.load(phase2_model_path, map_location=device))
    print(f"Loaded Phase 2 model successfully from {phase2_model_path}.\n")
    
    # FINAL CONFIGURATION
    NUM_WINDOWS = 5
    AGGREGATION = 'max'
    
    print(f"--- Running Final Test Config: {NUM_WINDOWS} Windows, {AGGREGATION.upper()} Aggregation ---")
    
    test_dataset = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=test_recs, num_windows=NUM_WINDOWS)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=multi_window_collate_fn)
    
    metrics = evaluate_multi_window(model, test_loader, device, aggregation=AGGREGATION)
    
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}\n")
    
    prefix = f"Phase3_TEST_{NUM_WINDOWS}Win_{AGGREGATION.capitalize()}"
    plot_curves(metrics['all_labels'], metrics['all_probs'], plot_dir, prefix)
    plot_confusion_matrix(metrics['cm'], plot_dir, prefix)
    
    counts = np.array(metrics['windows_counts'])
    
    # Write summary report
    with open(os.path.join(results_dir, "phase3_FINAL_TEST_results.txt"), "w") as f:
        f.write("==================================================\n")
        f.write("PHASE 3 FINAL BLIND TEST EVALUATION\n")
        f.write("==================================================\n\n")
        
        f.write(f"Model: Phase 2 ResNet-1D (No retraining)\n")
        f.write(f"Checkpoint Used: {phase2_model_path}\n")
        f.write(f"Windows Config: {NUM_WINDOWS}\n")
        f.write(f"Aggregation Config: {AGGREGATION.upper()}\n\n")
        
        f.write("--- Windows Statistics ---\n")
        f.write(f"Total TEST records: {len(test_recs)}\n")
        f.write(f"Total windows processed: {metrics['total_windows']}\n")
        f.write(f"Average windows per record: {np.mean(counts):.2f}\n")
        f.write(f"Runtime: {metrics['runtime']:.2f} seconds\n\n")
        
        f.write("--- Test Metrics vs Validation --- \n")
        f.write(f"{'Metric':<15} | {'Validation':<12} | {'Test'}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'ROC-AUC':<15} | {'0.8511':<12} | {metrics['roc_auc']:.4f}\n")
        f.write(f"{'PR-AUC':<15} | {'0.9614':<12} | {metrics['pr_auc']:.4f}\n")
        f.write(f"{'Accuracy':<15} | {'---':<12} | {metrics['accuracy']:.4f}\n")
        f.write(f"{'Sensitivity':<15} | {'---':<12} | {metrics['sensitivity']:.4f}\n")
        f.write(f"{'Specificity':<15} | {'---':<12} | {metrics['specificity']:.4f}\n")
        f.write(f"{'F1 Score':<15} | {'---':<12} | {metrics['f1']:.4f}\n\n")
        
        f.write(f"--- Detailed Classification Output ---\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['cm']}\n\n")
        
        f.write("--- Final Conclusion ---\n")
        f.write("Phase 1-3 testing complete. The model configuration is now entirely frozen.\n")

    print(f"Test experiments complete! Summary saved to {os.path.join(results_dir, 'phase3_FINAL_TEST_results.txt')}")

if __name__ == "__main__":
    main()
