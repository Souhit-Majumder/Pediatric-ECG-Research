import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import random
import pandas as pd

# Add Week6 to path to import dataset and model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Week6")))
from dataset import PediatricECGMultiWindowDataset, multi_window_collate_fn
from models import ResNet1D
from torch.utils.data import DataLoader

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
    
    test_patients = set(unique_patients[n_train+n_val:])
    
    test_records = []
    for pid, pr in patient_to_records.items():
        if pid in test_patients:
            test_records.extend(pr)
            
    return test_records

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, x, class_idx=1):
        self.model.eval()
        self.model.zero_grad()
        
        x.requires_grad = True
        
        output = self.model(x)
        score = output[0, class_idx]
        score.backward()
        
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)
        
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
            
        cam = cam.cpu().numpy()
        orig_length = x.shape[2]
        cam = np.interp(np.linspace(0, len(cam)-1, orig_length), np.arange(len(cam)), cam)
        
        return cam

def batch_generate_gradcam():
    print("Initializing Batch Grad-CAM generation...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    test_recs = patient_level_split(data_dir)
    print(f"Loaded {len(test_recs)} test records.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=2).to(device)
    
    checkpoint_path = os.path.join(project_root, "NoteBooks/Week6/Phase2/checkpoints/phase2_resnet_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 1. Run inference on the entire test set to categorize predictions
    dataset = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=test_recs, num_windows=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=multi_window_collate_fn)
    
    model.eval()
    results = []
    
    idx_tracker = 0
    with torch.no_grad():
        for batch_windows, batch_labels, counts in tqdm(dataloader, desc="Scoring Test Set"):
            batch_windows = batch_windows.to(device)
            outputs = model(batch_windows)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            c_idx = 0
            for label, c in zip(batch_labels, counts):
                record_probs = probs[c_idx:c_idx+c]
                max_prob_val, max_prob_idx = torch.max(record_probs, dim=0)
                
                results.append({
                    'record_path': test_recs[idx_tracker],
                    'label': label.item(),
                    'prob': max_prob_val.item(),
                    'max_window_idx': max_prob_idx.item(),
                    'pred': 1 if max_prob_val.item() >= 0.5 else 0
                })
                
                c_idx += c
                idx_tracker += 1
                
    df = pd.DataFrame(results)
    
    # Categorize
    df_tp = df[(df['label'] == 1) & (df['pred'] == 1)].sort_values('prob', ascending=False)
    df_tn = df[(df['label'] == 0) & (df['pred'] == 0)].sort_values('prob', ascending=True)
    df_fp = df[(df['label'] == 0) & (df['pred'] == 1)].sort_values('prob', ascending=False)
    df_fn = df[(df['label'] == 1) & (df['pred'] == 0)].sort_values('prob', ascending=True)
    
    # Select Top 10 representative cases for each
    representative_cases = {
        'TP': df_tp.head(10).to_dict('records'),
        'TN': df_tn.head(10).to_dict('records'),
        'FP': df_fp.head(10).to_dict('records'),
        'FN': df_fn.head(10).to_dict('records')
    }
    
    # Save the metadata for these cases
    results_dir = os.path.join(current_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame([item for sublist in representative_cases.values() for item in sublist]).to_csv(os.path.join(results_dir, "case_metadata.csv"), index=False)
    
    target_layer = model.layer3[-1].conv2
    grad_cam = GradCAM1D(model, target_layer)
    
    figures_dir = os.path.join(current_dir, "../figures")
    for category in ['TP', 'TN', 'FP', 'FN']:
        os.makedirs(os.path.join(figures_dir, category), exist_ok=True)
        
    print("\nGenerating Grad-CAM for representative cases...")
    for category, cases in representative_cases.items():
        print(f"Processing 10 cases for {category}...")
        for case in cases:
            rec_path = case['record_path']
            rec_name = os.path.basename(rec_path)
            
            # Load this specific record
            single_ds = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=[rec_path], num_windows=5)
            windows_tensor, label_tensor = single_ds[0]
            windows_tensor = windows_tensor.to(device)
            
            max_idx = case['max_window_idx']
            selected_window = windows_tensor[max_idx:max_idx+1]
            
            cam = grad_cam.generate(selected_window, class_idx=1)
            
            # Plot
            window_data = selected_window[0].cpu().detach().numpy()
            fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
            fig.suptitle(f"{category} | Record: {rec_name} | True: {case['label']} | Prob: {case['prob']:.4f}", fontsize=16)
            
            time_axis = np.linspace(0, 10, 5000)
            for i in range(12):
                axes[i].plot(time_axis, window_data[i], color='black', linewidth=0.8)
                axes[i].imshow(cam[np.newaxis, :], cmap='jet', aspect='auto', alpha=0.3, 
                               extent=[time_axis[0], time_axis[-1], window_data[i].min(), window_data[i].max()])
                axes[i].set_ylabel(f"Lead {i+1}")
                
            axes[-1].set_xlabel("Time (seconds)")
            
            out_path = os.path.join(figures_dir, category, f"{category}_{rec_name}.png")
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.savefig(out_path)
            plt.close(fig)

    print("Grad-CAM batch generation complete!")

if __name__ == "__main__":
    batch_generate_gradcam()
