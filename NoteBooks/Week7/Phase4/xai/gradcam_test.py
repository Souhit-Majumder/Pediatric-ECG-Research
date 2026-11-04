import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add Week6 to path to import dataset and model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Week6")))
from dataset import PediatricECGMultiWindowDataset, multi_window_collate_fn
from models import ResNet1D
import glob
import random

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
        
        # Interpolate to match original length (5000)
        orig_length = x.shape[2]
        cam = np.interp(np.linspace(0, len(cam)-1, orig_length), np.arange(len(cam)), cam)
        
        return cam, torch.softmax(output, dim=1)[0, class_idx].item()

def test_gradcam_one_ecg():
    print("Initializing Grad-CAM test...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    data_dir = os.path.join(project_root, "Child_ECG", "Child_ecg")
    
    test_recs = patient_level_split(data_dir)
    print(f"Loaded {len(test_recs)} test records.")
    
    # Pick the first record as a test
    test_rec = test_recs[0]
    
    dataset = PediatricECGMultiWindowDataset(data_dir=data_dir, apply_filter=False, record_paths=[test_rec], num_windows=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=2).to(device)
    
    checkpoint_path = os.path.join(project_root, "NoteBooks/Week6/Phase2/checkpoints/phase2_resnet_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # The final conv layer before GAP in ResNet1D is layer3's last block's conv2
    target_layer = model.layer3[-1].conv2
    grad_cam = GradCAM1D(model, target_layer)
    
    windows_tensor, label_tensor = dataset[0]
    windows_tensor = windows_tensor.to(device)
    
    # 1. Forward pass all 5 windows to find the MAX probability window
    model.eval()
    with torch.no_grad():
        outputs = model(windows_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1]
    
    max_idx = torch.argmax(probs).item()
    max_prob = probs[max_idx].item()
    
    print(f"Record: {os.path.basename(test_rec)}")
    print(f"True Label: {label_tensor.item()}")
    print(f"Number of distinct windows extracted: {windows_tensor.shape[0]}")
    print(f"Probabilities per window: {probs.cpu().numpy()}")
    print(f"Selected MAX window index: {max_idx}")
    print(f"MAX Probability: {max_prob:.4f}")
    
    # 2. Apply Grad-CAM to the selected MAX window
    selected_window = windows_tensor[max_idx:max_idx+1] # Keep batch dimension [1, 12, 5000]
    
    cam, cam_prob = grad_cam.generate(selected_window, class_idx=1) # Class 1 is CVD
    
    print(f"Grad-CAM generated. Shape: {cam.shape}, Min: {cam.min():.4f}, Max: {cam.max():.4f}")
    
    # Plotting 
    window_data = selected_window[0].cpu().detach().numpy()
    
    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f"Grad-CAM Test | Record: {os.path.basename(test_rec)} | True: {label_tensor.item()} | Prob: {max_prob:.4f}", fontsize=16)
    
    time_axis = np.linspace(0, 10, 5000)
    
    for i in range(12):
        axes[i].plot(time_axis, window_data[i], color='black', linewidth=0.8)
        
        # Color attribution
        axes[i].imshow(cam[np.newaxis, :], cmap='jet', aspect='auto', alpha=0.3, 
                       extent=[time_axis[0], time_axis[-1], window_data[i].min(), window_data[i].max()])
        axes[i].set_ylabel(f"Lead {i+1}")
        
    axes[-1].set_xlabel("Time (seconds)")
    
    out_path = os.path.join(current_dir, "../figures/gradcam_test_output.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    test_gradcam_one_ecg()
