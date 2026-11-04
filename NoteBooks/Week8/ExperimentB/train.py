import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import sys
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../NoteBooks/Week6")))
from dataset import PediatricECGDataset

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
            
    print(f"Patient Split -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    print(f"Record Split  -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
    
    return train_records, val_records, test_records

def main():
    print(f"\n{'='*50}\nStarting Week 8: Multi-scale ResNet-1D (Experiment B)\n{'='*50}")
    set_seed(42)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    plot_dir = os.path.join(current_dir, "plots")
    results_dir = os.path.join(current_dir, "results")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Configurable Hyperparameters (identical to Phase 2)
    max_epochs = 40
    early_stopping_patience = 6
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 1e-4
    
    # 1. Patient-Level Splitting
    train_recs, val_recs, test_recs = patient_level_split(data_dir)
    
    # 2. Datasets & Loaders
    train_dataset = PediatricECGDataset(data_dir=data_dir, apply_filter=False, record_paths=train_recs, random_crop=True)
    val_dataset = PediatricECGDataset(data_dir=data_dir, apply_filter=False, record_paths=val_recs, random_crop=False)
    
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 4 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    # 3. Class Weighting (Calculated ONLY on Training Set)
    labels_arr = np.array(train_dataset.labels, dtype=int)
    class_counts = np.bincount(labels_arr)
    weights = len(labels_arr) / (2.0 * class_counts)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    class_weights = torch.FloatTensor(weights).to(device)
    
    # 4. Model, Optimizer, Scheduler, Loss
    model = MultiScaleResNet1D(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Tracking
    best_val_auc = 0.0
    best_val_pr_auc = 0.0
    patience_counter = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_pr_auc': []}
    best_model_path = os.path.join(checkpoint_dir, "exp_b_multiscale_best.pth")
    
    # 5. Training Loop
    for epoch in range(max_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        epoch_preds, epoch_labels, epoch_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]  "):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs, 1)
                
                epoch_preds.extend(predicted.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
                epoch_probs.extend(probs.cpu().numpy())
                
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_auc = roc_auc_score(epoch_labels, epoch_probs)
        epoch_val_pr_auc = average_precision_score(epoch_labels, epoch_probs)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_auc'].append(epoch_val_auc)
        history['val_pr_auc'].append(epoch_val_pr_auc)
        
        print(f"Epoch [{epoch+1:02d}/{max_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {epoch_val_auc:.4f} | Val PR-AUC: {epoch_val_pr_auc:.4f}")
        
        # Scheduler Step
        scheduler.step(epoch_val_auc)
        
        # Early Stopping & Checkpointing
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_val_pr_auc = epoch_val_pr_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Val AUC! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}. Best Epoch was {best_epoch} with Val AUC: {best_val_auc:.4f}")
            break
            
    # 6. Plotting Curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs_range, history['val_auc'], 'g--', label='Val ROC-AUC')
    ax2.plot(epochs_range, history['val_pr_auc'], 'm--', label='Val PR-AUC')
    ax2.set_ylabel('AUC', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')
    
    plt.title('Experiment B: Multi-scale ResNet-1D')
    plt.axvline(x=best_epoch, color='k', linestyle=':', label=f'Best Epoch ({best_epoch})')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ExperimentB_Learning_Curves.png"))
    
    with open(os.path.join(results_dir, "experiment_b_val_results.txt"), "w") as f:
        f.write(f"Experiment B Best Epoch: {best_epoch}\n")
        f.write(f"Experiment B Validation ROC-AUC: {best_val_auc:.4f}\n")
        f.write(f"Experiment B Validation PR-AUC: {best_val_pr_auc:.4f}\n")
        
    print(f"\nTraining complete! Results saved to {results_dir}")

if __name__ == "__main__":
    main()
