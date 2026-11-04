import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from wfdb.processing import xqrs_detect

# Import existing dataset and model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../NoteBooks/Week6")))
from dataset import PediatricECGMultiWindowDataset
from models import ResNet1D

# Import existing GradCAM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../NoteBooks/Week7/Phase4/xai")))
from gradcam_batch import GradCAM1D

def detect_qrs(ecg_signal, fs=500):
    # Detect QRS using wfdb on Lead II (index 1) if possible
    # the signal shape is (num_leads, 5000)
    lead_idx = 1 if ecg_signal.shape[0] > 1 else 0
    sig = ecg_signal[lead_idx]
    
    # xqrs_detect expects 1D signal
    qrs_inds = xqrs_detect(sig=sig, fs=fs, verbose=False)
    return qrs_inds

def calculate_qrs_attribution(cam, qrs_inds, window_size_ms=150, fs=500):
    """
    cam: 1D array of Grad-CAM attribution
    qrs_inds: list of QRS peak indices
    window_size_ms: width of the window around each QRS peak to consider 'QRS region'
    """
    total_samples = len(cam)
    qrs_mask = np.zeros(total_samples, dtype=bool)
    
    half_window = int((window_size_ms / 2.0) * (fs / 1000.0))
    
    for idx in qrs_inds:
        start = max(0, idx - half_window)
        end = min(total_samples, idx + half_window)
        qrs_mask[start:end] = True
        
    qrs_attribution = np.sum(cam[qrs_mask])
    total_attribution = np.sum(cam)
    
    if total_attribution == 0:
        return 0.0
        
    return qrs_attribution / total_attribution

def main():
    print("Initializing QRS Attribution Analysis...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    meta_csv = os.path.join(project_root, "NoteBooks/Week7/Phase4/results/case_metadata.csv")
    if not os.path.exists(meta_csv):
        print(f"Error: {meta_csv} not found.")
        return
        
    df = pd.read_csv(meta_csv)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=2).to(device)
    checkpoint_path = os.path.join(project_root, "NoteBooks/Week6/Phase2/checkpoints/phase2_resnet_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    target_layer = model.layer3[-1].conv2
    grad_cam = GradCAM1D(model, target_layer)
    
    results = []
    
    # Map ecg_id back to actual file path since case_metadata contains 'record_path' which is absolute
    # However we might need to update paths if the project root has changed
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing QRS Coverage"):
        rec_path = row['record_path']
        label = row['label']
        pred = row['pred']
        
        # Categorize
        if label == 1 and pred == 1: category = 'TP'
        elif label == 0 and pred == 0: category = 'TN'
        elif label == 0 and pred == 1: category = 'FP'
        else: category = 'FN'
        
        max_idx = int(row['max_window_idx'])
        
        ds = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=[rec_path], num_windows=5)
        try:
            windows_tensor, _ = ds[0]
        except Exception as e:
            continue
            
        windows_tensor = windows_tensor.to(device)
        selected_window = windows_tensor[max_idx:max_idx+1]
        
        cam = grad_cam.generate(selected_window, class_idx=1)
        window_data = selected_window[0].cpu().detach().numpy()
        
        # QRS Detection
        qrs_inds = detect_qrs(window_data, fs=500)
        
        # Calculate attribution
        # We consider a QRS region to be 150ms wide (75ms each side of the peak)
        qrs_frac = calculate_qrs_attribution(cam, qrs_inds, window_size_ms=150, fs=500)
        
        results.append({
            'record': os.path.basename(rec_path),
            'category': category,
            'qrs_fraction': qrs_frac * 100 # percentage
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("QRS Attribution Coverage by Category")
    print("="*50)
    
    summary = res_df.groupby('category')['qrs_fraction'].mean().reset_index()
    # Sort TP, TN, FP, FN
    order = {'TP': 1, 'TN': 2, 'FP': 3, 'FN': 4}
    summary['order'] = summary['category'].map(order)
    summary = summary.sort_values('order').drop('order', axis=1)
    
    for _, row in summary.iterrows():
        print(f"{row['category']:<5} {row['qrs_fraction']:.1f}%")
        
    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    res_df.to_csv(os.path.join(out_dir, "qrs_attribution_stats.csv"), index=False)
    print(f"\nSaved detailed results to {out_dir}/qrs_attribution_stats.csv")

if __name__ == "__main__":
    main()
