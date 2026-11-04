import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from wfdb.processing import xqrs_detect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../NoteBooks/Week6")))
from dataset import PediatricECGMultiWindowDataset
from models import ResNet1D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../NoteBooks/Week7/Phase4/xai")))
from gradcam_batch import GradCAM1D

def detect_qrs_multi_lead(ecg_signal, fs=500):
    num_leads = ecg_signal.shape[0]
    # Check Lead II (index 1) and a precordial lead, e.g., V5 (index 10) or last lead if 9-lead
    leads_to_check = [1]
    if num_leads > 10:
        leads_to_check.append(10) # V5
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
            
    # Remove duplicates within 50ms (25 samples)
    if not all_peaks:
        return []
        
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
    print("Initializing Multi-Lead Validated QRS Attribution Analysis...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    meta_csv = os.path.join(project_root, "NoteBooks/Week7/Phase4/results/case_metadata.csv")
    df = pd.read_csv(meta_csv)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=2).to(device)
    checkpoint_path = os.path.join(project_root, "NoteBooks/Week6/Phase2/checkpoints/phase2_resnet_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    target_layer = model.layer3[-1].conv2
    grad_cam = GradCAM1D(model, target_layer)
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing QRS Coverage (Multi-lead)"):
        rec_path = row['record_path']
        label = row['label']
        pred = row['pred']
        
        if label == 1 and pred == 1: category = 'TP'
        elif label == 0 and pred == 0: category = 'TN'
        elif label == 0 and pred == 1: category = 'FP'
        else: category = 'FN'
        
        max_idx = int(row['max_window_idx'])
        
        ds = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=[rec_path], num_windows=5)
        try:
            windows_tensor, _ = ds[0]
        except Exception:
            continue
            
        windows_tensor = windows_tensor.to(device)
        selected_window = windows_tensor[max_idx:max_idx+1]
        
        cam = grad_cam.generate(selected_window, class_idx=1)
        window_data = selected_window[0].cpu().detach().numpy()
        
        qrs_inds = detect_qrs_multi_lead(window_data, fs=500)
        qrs_frac = calculate_qrs_attribution(cam, qrs_inds, window_size_ms=150, fs=500)
        
        results.append({
            'record': os.path.basename(rec_path),
            'category': category,
            'qrs_fraction': qrs_frac * 100
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("VALIDATED QRS Attribution Coverage by Category")
    print("="*50)
    
    summary = res_df.groupby('category')['qrs_fraction'].mean().reset_index()
    order = {'TP': 1, 'TN': 2, 'FP': 3, 'FN': 4}
    summary['order'] = summary['category'].map(order)
    summary = summary.sort_values('order').drop('order', axis=1)
    
    for _, row in summary.iterrows():
        print(f"{row['category']:<5} {row['qrs_fraction']:.1f}%")

if __name__ == "__main__":
    main()
