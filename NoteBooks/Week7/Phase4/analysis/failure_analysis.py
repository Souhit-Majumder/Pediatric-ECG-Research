import os
import sys
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Week6")))
from dataset import PediatricECGMultiWindowDataset, multi_window_collate_fn
from models import ResNet1D

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

def get_age_group(years):
    if years < 1/12: return '01_Neonate (<1m)'
    if years < 1: return '02_Infant (1m-1y)'
    if years < 3: return '03_Toddler (1-3y)'
    if years < 5: return '04_Preschool (3-5y)'
    if years < 12: return '05_School (5-12y)'
    return '06_Adolescent (12-18y)'

def main():
    print("Initializing Failure Analysis...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    test_recs = patient_level_split(data_dir)
    print(f"Loaded {len(test_recs)} test records.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=2).to(device)
    
    checkpoint_path = os.path.join(project_root, "NoteBooks/Week6/Phase2/checkpoints/phase2_resnet_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load Attributes Dictionary
    attr_path = os.path.join(project_root, "Data", "AttributesDictionary.csv")
    attr_df = pd.read_csv(attr_path)
    attr_df['ECG_ID'] = attr_df['ECG_ID'].astype(str)
    
    dataset = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=test_recs, num_windows=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=multi_window_collate_fn)
    
    results = []
    idx_tracker = 0
    
    with torch.no_grad():
        for batch_windows, batch_labels, counts in tqdm(dataloader, desc="Scoring Test Set & Merging Metadata"):
            batch_windows = batch_windows.to(device)
            outputs = model(batch_windows)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            c_idx = 0
            for label, c in zip(batch_labels, counts):
                record_probs = probs[c_idx:c_idx+c]
                max_prob_val, max_prob_idx = torch.max(record_probs, dim=0)
                
                rec_path = test_recs[idx_tracker]
                ecg_id = os.path.basename(rec_path)
                
                # Fetch metadata
                row = attr_df[attr_df['ECG_ID'] == ecg_id]
                if len(row) > 0:
                    row = row.iloc[0]
                    age_str = str(row['Age'])
                    age_days = float(age_str.replace('d', '')) if 'd' in age_str else float(age_str)
                    age_years = age_days / 365.25
                    sex = str(row['Gender']).replace("'", "")
                    num_leads = row['Lead']
                    duration = row['Sampling_point'] / 500.0
                    disease_label = str(row['ICD-10 code']).replace("'", "")
                    ecg_diag = str(row['AHA_code']).replace("'", "")
                    patient_id = row['Patient_ID']
                else:
                    age_days, age_years, sex, num_leads, duration, disease_label, ecg_diag, patient_id = 0, 0, "Unknown", 12, 10, "Unknown", "Unknown", "Unknown"
                
                results.append({
                    'patient_id': patient_id,
                    'ecg_id': ecg_id,
                    'true_label': label.item(),
                    'predicted_label': 1 if max_prob_val.item() >= 0.5 else 0,
                    'prediction_probability': max_prob_val.item(),
                    'age_days': age_days,
                    'age_years': age_years,
                    'age_group': get_age_group(age_years),
                    'sex': sex,
                    'lead_configuration': f"{num_leads}-lead",
                    'num_leads': num_leads,
                    'duration_seconds': duration,
                    'disease_label': disease_label,
                    'ecg_diagnostic_statement': ecg_diag,
                    'selected_window': max_prob_idx.item()
                })
                
                c_idx += c
                idx_tracker += 1

    df = pd.DataFrame(results)
    
    # Filter FNs
    fn_df = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)].copy()
    
    # Save the FN CSV
    out_dir = os.path.join(current_dir, "../results")
    os.makedirs(out_dir, exist_ok=True)
    fn_csv_path = os.path.join(out_dir, "failure_analysis.csv")
    fn_df.to_csv(fn_csv_path, index=False)
    print(f"\nSaved FN metadata ({len(fn_df)} records) to {fn_csv_path}")
    
    # We also want to compute statistics for Age, Leads, Disease
    # 1. Age Group
    print("\n" + "="*50)
    print("1. Age Group Analysis")
    print("="*50)
    age_stats = []
    for g in sorted(df['age_group'].unique()):
        g_total = len(df[df['age_group'] == g])
        g_fn = len(fn_df[fn_df['age_group'] == g])
        fn_rate = (g_fn / g_total * 100) if g_total > 0 else 0
        age_stats.append({'Age group': g, 'ECGs': g_total, 'FNs': g_fn, 'FN rate (%)': fn_rate})
    age_df = pd.DataFrame(age_stats)
    print(age_df.to_string(index=False))

    # 2. Lead Configuration
    print("\n" + "="*50)
    print("2. Lead Configuration Analysis")
    print("="*50)
    lead_stats = []
    for lc in sorted(df['lead_configuration'].unique()):
        lc_total = len(df[df['lead_configuration'] == lc])
        lc_fn = len(fn_df[fn_df['lead_configuration'] == lc])
        fn_rate = (lc_fn / lc_total * 100) if lc_total > 0 else 0
        lead_stats.append({'Configuration': lc, 'ECGs': lc_total, 'FNs': lc_fn, 'FN rate (%)': fn_rate})
    lead_df = pd.DataFrame(lead_stats)
    print(lead_df.to_string(index=False))
    
    # Helper to parse multi-labels
    def explode_labels(dataframe, col_name):
        expanded = []
        for _, row in dataframe.iterrows():
            labels = str(row[col_name]).split(';')
            for l in labels:
                if pd.isna(l) or l.strip() == '' or l == 'nan': continue
                new_row = row.copy()
                new_row[col_name] = l.strip()
                expanded.append(new_row)
        return pd.DataFrame(expanded)

    # 3. Disease Label (ICD-10)
    print("\n" + "="*50)
    print("3. Disease Label (ICD-10) Analysis")
    print("="*50)
    df_dis = explode_labels(df, 'disease_label')
    fn_dis = explode_labels(fn_df, 'disease_label')
    dis_stats = []
    for d in sorted(df_dis['disease_label'].unique()):
        d_total = len(df_dis[df_dis['disease_label'] == d])
        d_fn = len(fn_dis[fn_dis['disease_label'] == d])
        fn_rate = (d_fn / d_total * 100) if d_total > 0 else 0
        if d_total > 5: # Only show diseases with > 5 cases in test set to avoid extreme noise
            dis_stats.append({'Disease label': d, 'ECGs': d_total, 'FNs': d_fn, 'FN rate (%)': fn_rate})
    dis_df = pd.DataFrame(dis_stats).sort_values('FN rate (%)', ascending=False)
    print(dis_df.head(20).to_string(index=False))
    print("... (showing top 20 by FN rate for diseases with >5 samples)")
    
    # 4. ECG Diagnostic Statement (AHA)
    print("\n" + "="*50)
    print("4. ECG Diagnostic Statement (AHA Code) Analysis")
    print("="*50)
    df_ecg = explode_labels(df, 'ecg_diagnostic_statement')
    fn_ecg = explode_labels(fn_df, 'ecg_diagnostic_statement')
    ecg_stats = []
    for e in sorted(df_ecg['ecg_diagnostic_statement'].unique()):
        e_total = len(df_ecg[df_ecg['ecg_diagnostic_statement'] == e])
        e_fn = len(fn_ecg[fn_ecg['ecg_diagnostic_statement'] == e])
        fn_rate = (e_fn / e_total * 100) if e_total > 0 else 0
        if e_total > 5:
            ecg_stats.append({'Diagnostic Statement': e, 'ECGs': e_total, 'FNs': e_fn, 'FN rate (%)': fn_rate})
    ecg_df = pd.DataFrame(ecg_stats).sort_values('FN rate (%)', ascending=False)
    print(ecg_df.head(20).to_string(index=False))
    print("... (showing top 20 by FN rate for diagnoses with >5 samples)")
    
    # 5. Prediction Probability Distribution
    print("\n" + "="*50)
    print("5. False Negative Probability Distribution")
    print("="*50)
    borderline = len(fn_df[(fn_df['prediction_probability'] >= 0.40) & (fn_df['prediction_probability'] < 0.50)])
    confident = len(fn_df[fn_df['prediction_probability'] < 0.10])
    mid = len(fn_df[(fn_df['prediction_probability'] >= 0.10) & (fn_df['prediction_probability'] < 0.40)])
    
    print(f"Total False Negatives: {len(fn_df)}")
    print(f"Borderline FNs (Prob 0.40 - 0.49): {borderline} ({borderline/len(fn_df)*100:.1f}%)")
    print(f"Mid-confidence FNs (Prob 0.10 - 0.39): {mid} ({mid/len(fn_df)*100:.1f}%)")
    print(f"Confident FNs (Prob < 0.10): {confident} ({confident/len(fn_df)*100:.1f}%)")
    
if __name__ == "__main__":
    main()
