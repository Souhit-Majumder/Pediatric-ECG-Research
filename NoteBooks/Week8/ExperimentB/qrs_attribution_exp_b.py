import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from wfdb.processing import xqrs_detect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../NoteBooks/Week6")))
from dataset import PediatricECGMultiWindowDataset

from models import MultiScaleResNet1D

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

def detect_qrs_multi_lead(ecg_signal, fs=500):
    num_leads = ecg_signal.shape[0]
    leads_to_check = [1]
    if num_leads > 10:
        leads_to_check.append(10)
    else:
        leads_to_check.append(num_leads - 1)
        
    all_peaks = []
    for l_idx in leads_to_check:
        sig = ecg_signal[l_idx]
        try:
            inds = xqrs_detect(sig=sig, fs=fs, verbose=False)
            all_peaks.extend(inds)
        except Exception:
            continue
            
    if not all_peaks: return []
        
    all_peaks.sort()
    merged_peaks = [all_peaks[0]]
    for p in all_peaks[1:]:
        if p - merged_peaks[-1] > 25:
            merged_peaks.append(p)
            
    return merged_peaks

def calculate_qrs_attribution(cam, qrs_inds, window_size_ms=150, fs=500):
    total_samples = len(cam)
    qrs_mask = np.zeros(total_samples, dtype=bool)
    half_window = int((window_size_ms / 2.0) * (fs / 1000.0))
    
    for idx in qrs_inds:
        start = max(0, idx - half_window)
        end = min(total_samples, idx + half_window)
        qrs_mask[start:end] = True
        
    qrs_attribution = np.sum(cam[qrs_mask])
    total_attribution = np.sum(cam)
    
    if total_attribution == 0: return 0.0
    return qrs_attribution / total_attribution

def main():
    print("Initializing Multi-Scale QRS Attribution Analysis...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    # We evaluate on the SAME 40 cases as the baseline model for an exact comparison
    meta_csv = os.path.join(project_root, "NoteBooks/Week7/Phase4/results/case_metadata.csv")
    if not os.path.exists(meta_csv):
        print("Could not find the 40 cases metadata from Week 7.")
        return
    df = pd.read_csv(meta_csv)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleResNet1D(num_classes=2).to(device)
    checkpoint_path = os.path.join(current_dir, "checkpoints", "exp_b_multiscale_best.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure training is complete before running attribution analysis.")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # For MultiScaleResNet1D, the last conv layer is in layer3[-1].conv2.branch1/2/3
    # Let's target the final bottleneck before maxpool, or layer3[-1].bn2 (the final batchnorm of the residual block)
    target_layer = model.layer3[-1].bn2
    grad_cam = GradCAM1D(model, target_layer)
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Multi-Scale QRS Coverage"):
        rec_path = row['record_path']
        # Note: the true label stays the same, but the model's prediction might change.
        # We will re-evaluate the prediction using the new model.
        label = row['label']
        
        ds = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=[rec_path], num_windows=5)
        try:
            windows_tensor, _ = ds[0]
        except Exception:
            continue
            
        windows_tensor = windows_tensor.to(device)
        
        # We find the new MAX window for this model
        with torch.no_grad():
            model.eval()
            outputs = model(windows_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            max_idx = torch.argmax(probs).item()
            pred = 1 if probs[max_idx].item() >= 0.5 else 0
            
        selected_window = windows_tensor[max_idx:max_idx+1]
        cam = grad_cam.generate(selected_window, class_idx=1)
        window_data = selected_window[0].cpu().detach().numpy()
        
        qrs_inds = detect_qrs_multi_lead(window_data, fs=500)
        qrs_frac = calculate_qrs_attribution(cam, qrs_inds, window_size_ms=150, fs=500)
        
        # Categorize based on NEW predictions
        if label == 1 and pred == 1: category = 'TP'
        elif label == 0 and pred == 0: category = 'TN'
        elif label == 0 and pred == 1: category = 'FP'
        else: category = 'FN'
        
        results.append({
            'record': os.path.basename(rec_path),
            'category': category,
            'qrs_fraction': qrs_frac * 100
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("Multi-Scale QRS Attribution Coverage by Category")
    print("="*50)
    
    summary = res_df.groupby('category')['qrs_fraction'].mean().reset_index()
    order = {'TP': 1, 'TN': 2, 'FP': 3, 'FN': 4}
    summary['order'] = summary['category'].map(order)
    summary = summary.sort_values('order').drop('order', axis=1)
    
    for _, row in summary.iterrows():
        print(f"{row['category']:<5} {row['qrs_fraction']:.1f}%")
        
    out_dir = os.path.join(current_dir, "results")
    res_df.to_csv(os.path.join(out_dir, "exp_b_qrs_attribution_stats.csv"), index=False)
    print(f"\nSaved detailed results to {out_dir}/exp_b_qrs_attribution_stats.csv")

if __name__ == "__main__":
    main()
