import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wfdb
import glob
import os
import numpy as np

# Simple 1D CNN for ECG processing
class SimpleECGModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleECGModel, self).__init__()
        # 12 input channels (standard 12-lead ECG), 16 output channels
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # We will dynamically calculate the fully connected layer input size
        # by passing a dummy tensor during initialization.
        self.fc1 = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class PediatricECGDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5000):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        # Find all .hea files
        self.records = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
        # Strip the .hea extension to get the record name for wfdb
        self.records = [r[:-4] for r in self.records]
        
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        
        # Read the record using wfdb
        # rdsamp reads both the .dat and .hea files
        signals, fields = wfdb.rdsamp(record_path)
        
        # signals is shape (samples, channels) e.g., (15000, 12)
        # PyTorch Conv1d expects (channels, samples)
        signals = np.transpose(signals)
        
        # 1) Truncate or pad CHANNELS to 12 (standard 12-lead)
        target_channels = 12
        if signals.shape[0] > target_channels:
            signals = signals[:target_channels, :]
        elif signals.shape[0] < target_channels:
            pad_channels = target_channels - signals.shape[0]
            signals = np.pad(signals, ((0, pad_channels), (0, 0)), mode='constant')
        
        # 2) Truncate or pad SEQUENCE LENGTH to ensure consistent tensor sizes
        if signals.shape[1] > self.sequence_length:
            signals = signals[:, :self.sequence_length]
        elif signals.shape[1] < self.sequence_length:
            pad_width = self.sequence_length - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
            
        # Extract label from the comments in fields
        comments = fields.get('comments', [])
        diagnosis = comments[-1].lower() if comments else ""
        
        # Simple binary labeling logic based on Week 3:
        # "normal ecg" -> 0 (Non-CVD), everything else -> 1 (CVD)
        if "normal ecg" in diagnosis:
            label = 0
        else:
            label = 1
            
        # Convert to PyTorch tensors
        signal_tensor = torch.tensor(signals, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor

def main():
    print("Setting up Week 4 PyTorch Model with Real Data...")
    
    # Path to the dataset (relative to the script execution point)
    data_dir = "Child_ECG/Child_ecg/"
    
    # Example hyperparameters
    batch_size = 16
    sequence_length = 5000 # Typical 10-second ECG at 500Hz is 5000 samples
    num_classes = 2 # Normal vs CVD
    num_epochs = 5
    
    print(f"Loading dataset from {data_dir}...")
    dataset = PediatricECGDataset(data_dir=data_dir, sequence_length=sequence_length)
    
    if len(dataset) == 0:
        print("Error: No records found. Please check the data_dir path.")
        return
        
    print(f"Found {len(dataset)} ECG records.")
    
    # PyTorch native Train/Test Split (80% Train, 20% Test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print(f"Split dataset: {train_size} training records, {test_size} testing records.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = SimpleECGModel(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Import roc_auc_score from sklearn (computing AUC from scratch in pure PyTorch is very complex)
    from sklearn.metrics import roc_auc_score
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print less frequently to avoid terminal spam
            if (i+1) % 100 == 0:
                print(f"  Train Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")
                
        # --- EVALUATION PHASE ---
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Disable gradient calculation for evaluation to save memory and speed up
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Get probabilities for the positive class (CVD = 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs, 1)
                
                all_preds.append(predicted)
                all_labels.append(labels)
                all_probs.append(probs)
                
        # Concatenate lists of tensors into single tensors
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        
        # 1. Pure PyTorch Accuracy
        correct = (all_preds == all_labels).sum().item()
        accuracy = 100 * correct / len(all_labels)
        
        # 2. Pure PyTorch Precision & Recall
        # tp = True Positives, fp = False Positives, fn = False Negatives
        tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 3. ROC-AUC (Using sklearn since pure PyTorch implementation requires external libraries like torchmetrics)
        try:
            roc_auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
        except ValueError:
            roc_auc = 0.5 # Fallback if test set somehow only contains one class
            
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"  Test Loss:  {test_loss/len(test_loader):.4f}")
        print(f"  Accuracy:   {accuracy:.2f}%")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  ROC-AUC:    {roc_auc:.4f}\n")
        
    print("Training and Evaluation complete!")

if __name__ == "__main__":
    main()
