import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve

from dataset import PediatricECGDataset
from models import SimpleECGModel, ResNet1D

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_and_evaluate(experiment_name, model_class, apply_filter, data_dir, batch_size=16, num_epochs=5):
    print(f"\n{'='*50}\nStarting {experiment_name}\n{'='*50}")
    set_seed(42) 
    
    dataset = PediatricECGDataset(data_dir=data_dir, sequence_length=5000, apply_filter=apply_filter)
    
    labels_arr = np.array(dataset.labels, dtype=int)
    class_counts = np.bincount(labels_arr)
    weights = len(labels_arr) / (2.0 * class_counts)
    class_weights = torch.FloatTensor(weights)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = model_class(num_classes=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
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
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs, 1)
                
                epoch_preds.extend(predicted.numpy())
                epoch_labels.extend(labels.numpy())
                epoch_probs.extend(probs.numpy())
                
        epoch_val_loss = running_val_loss / len(test_loader)
        epoch_val_auc = roc_auc_score(epoch_labels, epoch_probs)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_auc'].append(epoch_val_auc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {epoch_val_auc:.4f}")
    
    # Final evaluation metrics
    final_auc = history['val_auc'][-1]
    precision = precision_score(epoch_labels, epoch_preds, zero_division=0)
    recall = recall_score(epoch_labels, epoch_preds, zero_division=0)
    f1 = f1_score(epoch_labels, epoch_preds, zero_division=0)
    cm = confusion_matrix(epoch_labels, epoch_preds)
    
    print(f"\n{experiment_name} Final Results:")
    print(f"  ROC-AUC:   {final_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return history, epoch_labels, epoch_probs

def plot_curves(all_results, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Training vs Validation Loss and Val AUC (per model)
    for exp_name, res in all_results.items():
        hist = res['history']
        epochs = range(1, len(hist['train_loss']) + 1)
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, hist['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, hist['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.plot(epochs, hist['val_auc'], 'g--', label='Val ROC-AUC')
        ax2.set_ylabel('ROC-AUC', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper right')
        
        plt.title(f"{exp_name}: Training Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{exp_name.replace(' ', '_').replace(':', '')}_learning_curves.png"))
        plt.close()

    # 2. Combined Final ROC Curves
    plt.figure(figsize=(10, 8))
    for exp_name, res in all_results.items():
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        auc_val = res['history']['val_auc'][-1]
        plt.plot(fpr, tpr, lw=2, label=f"{exp_name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "Combined_ROC_Curves.png"))
    plt.close()

    # 3. Combined Final PR Curves
    plt.figure(figsize=(10, 8))
    for exp_name, res in all_results.items():
        precision, recall, thresholds = precision_recall_curve(res['labels'], res['probs'])
        plt.plot(recall, precision, lw=2, label=f"{exp_name}")
        
        # Threshold analysis (Find best F1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        print(f"{exp_name} Best Threshold: {best_threshold:.3f} (F1: {f1_scores[best_idx]:.3f})")
        plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='red')
        
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall (PR) Curves with Optimal F1 Thresholds')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "Combined_PR_Curves.png"))
    plt.close()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    plot_dir = os.path.join(project_root, "Docs", "Week5", "plots")
    
    # Proper Training: 5 epochs for all
    epochs = 5 
    
    all_results = {}
    
    # Run the 4 experiments
    hist, labels, probs = train_and_evaluate("Exp 1: Baseline CNN", SimpleECGModel, False, data_dir, num_epochs=epochs)
    all_results['E1_Raw_CNN'] = {'history': hist, 'labels': labels, 'probs': probs}
    
    hist, labels, probs = train_and_evaluate("Exp 2: Filtered CNN", SimpleECGModel, True, data_dir, num_epochs=epochs)
    all_results['E2_Filtered_CNN'] = {'history': hist, 'labels': labels, 'probs': probs}
    
    hist, labels, probs = train_and_evaluate("Exp 3: Raw ResNet1D", ResNet1D, False, data_dir, num_epochs=epochs)
    all_results['E3_Raw_ResNet'] = {'history': hist, 'labels': labels, 'probs': probs}
    
    hist, labels, probs = train_and_evaluate("Exp 4: Filtered ResNet1D", ResNet1D, True, data_dir, num_epochs=epochs)
    all_results['E4_Filtered_ResNet'] = {'history': hist, 'labels': labels, 'probs': probs}
    
    # Generate Curves and Threshold Analysis
    print(f"\n{'='*50}\nGenerating Curves and Threshold Analysis...\n{'='*50}")
    plot_curves(all_results, plot_dir)
    print(f"All plots successfully saved to {plot_dir}")

if __name__ == "__main__":
    main()
