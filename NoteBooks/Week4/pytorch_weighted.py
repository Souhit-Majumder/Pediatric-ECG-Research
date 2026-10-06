import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wfdb
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

class SimpleECGModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class PediatricECGDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5000):
        self.records = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
        self.records = [r[:-4] for r in self.records]
        self.sequence_length = sequence_length
        
        # Pre-calculate class distribution for weighting
        print("Scanning dataset to calculate class weights...")
        self.labels = []
        for r in self.records:
            _, fields = wfdb.rdsamp(r, sampto=1) # only read headers for speed
            comments = fields.get('comments', [])
            diagnosis = comments[-1].lower() if comments else ""
            self.labels.append(0 if "normal ecg" in diagnosis else 1)
            
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        signals, _ = wfdb.rdsamp(record_path)
        signals = np.transpose(signals)
        
        # Standardize Channels to 12
        target_channels = 12
        if signals.shape[0] > target_channels:
            signals = signals[:target_channels, :]
        elif signals.shape[0] < target_channels:
            pad_channels = target_channels - signals.shape[0]
            signals = np.pad(signals, ((0, pad_channels), (0, 0)), mode='constant')
            
        # Standardize Sequence Length
        if signals.shape[1] > self.sequence_length:
            signals = signals[:, :self.sequence_length]
        elif signals.shape[1] < self.sequence_length:
            pad_width = self.sequence_length - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
            
        return torch.tensor(signals, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def main():
    print("Setting up Week 4 Class-Weighted Model...")
    data_dir = "Child_ECG/Child_ecg/"
    batch_size = 16
    sequence_length = 5000
    num_epochs = 3
    
    dataset = PediatricECGDataset(data_dir=data_dir, sequence_length=sequence_length)
    
    # Calculate Class Weights
    labels_arr = np.array(dataset.labels)
    class_counts = np.bincount(labels_arr)
    total_samples = len(labels_arr)
    
    # Inverse frequency weighting
    weights = total_samples / (2.0 * class_counts)
    class_weights = torch.FloatTensor(weights)
    print(f"Class Distribution: Normal={class_counts[0]}, CVD={class_counts[1]}")
    print(f"Calculated Class Weights: Normal={weights[0]:.4f}, CVD={weights[1]:.4f}")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SimpleECGModel(num_classes=2)
    # Apply weights to Loss Function!
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting Training...")
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Batch {i+1} Loss: {loss.item():.4f}")
                
    print("\nTraining Complete! Evaluating on Test Set...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
            
    # Calculate Final Metrics
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nFinal Test ROC-AUC: {roc_auc:.4f}")
    
    # Generate Tangible Output Plots
    print("Generating Evaluation Plots...")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'CVD'], yticklabels=['Normal', 'CVD'])
    plt.title('Confusion Matrix (Weighted CNN)')
    plt.ylabel('True Diagnosis')
    plt.xlabel('Predicted Diagnosis')
    
    # Plot 2: ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save the plot
    os.makedirs("Docs/Week4", exist_ok=True)
    plot_path = "Docs/Week4/model_evaluation_weighted.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Success! Evaluation plots saved to: {plot_path}")

if __name__ == "__main__":
    main()
