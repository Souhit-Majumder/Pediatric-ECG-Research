import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import glob

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from gradcam_batch import GradCAM1D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Week6")))
from dataset import PediatricECGMultiWindowDataset
from models import ResNet1D

def select_representative_cases(df, n=10):
    if len(df) <= n:
        return df
    # sort by probability to cover the range
    df = df.sort_values('prediction_probability')
    indices = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[indices]

def main():
    print("Initializing Targeted FN Grad-CAM generation...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    fn_csv_path = os.path.join(current_dir, "../results/failure_analysis.csv")
    if not os.path.exists(fn_csv_path):
        print(f"Error: {fn_csv_path} not found.")
        return
        
    fn_df = pd.read_csv(fn_csv_path)
    
    # Identify subgroups
    # 1. Confident FNs (< 0.10)
    confident_df = fn_df[fn_df['prediction_probability'] < 0.10].copy()
    
    # 2. LV High Voltage
    lvh_df = fn_df[fn_df['ecg_diagnostic_statement'].fillna('').str.contains('Left ventricular high voltage', case=False, regex=False)].copy()
    
    # 3. Early repolarization (L155)
    l155_df = fn_df[fn_df['ecg_diagnostic_statement'].fillna('').str.contains('L155', case=False, regex=False)].copy()
    
    # 4. ST deviation (L145)
    l145_df = fn_df[fn_df['ecg_diagnostic_statement'].fillna('').str.contains('L145', case=False, regex=False)].copy()
    
    # 5. Other FN Control Group
    # Exclude all of the above
    exclude_ids = set(confident_df['ecg_id']) | set(lvh_df['ecg_id']) | set(l155_df['ecg_id']) | set(l145_df['ecg_id'])
    control_df = fn_df[~fn_df['ecg_id'].isin(exclude_ids)].copy()
    
    # Select representatives
    subgroups = {
        'Confident_FNs': confident_df,
        'LVH': select_representative_cases(lvh_df, 10),
        'Early_Repolarization_L155': select_representative_cases(l155_df, 10),
        'ST_Deviation_L145': select_representative_cases(l145_df, 10),
        'Control_Other_FNs': select_representative_cases(control_df, 10)
    }
    
    # Save the selected metadata
    selected_cases = []
    for group_name, df_subset in subgroups.items():
        for _, row in df_subset.iterrows():
            r = row.to_dict()
            r['subgroup'] = group_name
            selected_cases.append(r)
            
    selected_df = pd.DataFrame(selected_cases)
    out_meta_path = os.path.join(current_dir, "../results/targeted_fn_metadata.csv")
    selected_df.to_csv(out_meta_path, index=False)
    print(f"Saved selected cases metadata to {out_meta_path}")
    
    # Initialize Model & Grad-CAM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=2).to(device)
    checkpoint_path = os.path.join(project_root, "NoteBooks/Week6/Phase2/checkpoints/phase2_resnet_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    target_layer = model.layer3[-1].conv2
    grad_cam = GradCAM1D(model, target_layer)
    
    # Generate Grad-CAMs
    figures_base_dir = os.path.join(current_dir, "../figures/targeted_fns")
    os.makedirs(figures_base_dir, exist_ok=True)
    
    # Map ecg_id back to actual file path
    all_hea = glob.glob(os.path.join(data_dir, '**/*.hea'), recursive=True)
    ecg_id_to_path = {os.path.basename(p)[:-4]: p[:-4] for p in all_hea}
    
    for group_name, df_subset in subgroups.items():
        print(f"\nProcessing {len(df_subset)} cases for {group_name}...")
        group_dir = os.path.join(figures_base_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        for _, row in df_subset.iterrows():
            ecg_id = row['ecg_id']
            if ecg_id not in ecg_id_to_path:
                print(f"Warning: Could not find path for {ecg_id}")
                continue
                
            rec_path = ecg_id_to_path[ecg_id]
            max_idx = int(row['selected_window'])
            prob = row['prediction_probability']
            
            # Load
            single_ds = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=[rec_path], num_windows=5)
            try:
                windows_tensor, label_tensor = single_ds[0]
            except Exception as e:
                print(f"Error loading {ecg_id}: {e}")
                continue
                
            windows_tensor = windows_tensor.to(device)
            selected_window = windows_tensor[max_idx:max_idx+1]
            
            cam = grad_cam.generate(selected_window, class_idx=1)
            
            # Plot
            window_data = selected_window[0].cpu().detach().numpy()
            num_leads_in_data = window_data.shape[0]
            
            # Create dynamic subplots depending on lead count (usually 9 or 12)
            fig, axes = plt.subplots(num_leads_in_data, 1, figsize=(15, 1.5 * num_leads_in_data), sharex=True)
            if num_leads_in_data == 1:
                axes = [axes]
                
            fig.suptitle(f"Subgroup: {group_name} | Record: {ecg_id} | Prob: {prob:.4f}\nAge: {row.get('age_years', 0):.1f}y | ECG Diag: {str(row.get('ecg_diagnostic_statement', ''))[:100]}", fontsize=14)
            
            time_axis = np.linspace(0, 10, 5000)
            for i in range(num_leads_in_data):
                axes[i].plot(time_axis, window_data[i], color='black', linewidth=0.8)
                axes[i].imshow(cam[np.newaxis, :], cmap='jet', aspect='auto', alpha=0.3, 
                               extent=[time_axis[0], time_axis[-1], window_data[i].min(), window_data[i].max()])
                axes[i].set_ylabel(f"Lead {i+1}")
                
            axes[-1].set_xlabel("Time (seconds)")
            
            out_path = os.path.join(group_dir, f"{ecg_id}.png")
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.savefig(out_path)
            plt.close(fig)

    print("\nTargeted FN Grad-CAM generation complete!")

if __name__ == "__main__":
    main()
