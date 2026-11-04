# Phase 4 Model Verification

## Verified Configuration Details

1. **Exact model architecture**: `ResNet1D`. Initial Conv1D (12->32, k=15) -> MaxPool -> Layer1 (32->32, 2 blocks) -> Layer2 (32->64, 2 blocks) -> Layer3 (64->128, 2 blocks) -> Global Average Pooling -> Linear (128->2).
2. **Checkpoint loading**: `/home/souhit/Documents/Research/Pediatric_ECG/NoteBooks/Week6/Phase2/checkpoints/phase2_resnet_best.pth`.
3. **Input tensor shape**: `12 × 5000` (12 leads, 5000 samples = 10 seconds at 500Hz).
4. **Lead ordering**: Native `wfdb` extraction order (assumed standard 12-lead arrangement, no explicit reordering step present).
5. **Normalization/preprocessing**: Z-score normalization per lead. Signals are padded with zeros if shorter than 5000 or < 12 channels. Bandpass filtering is disabled (`apply_filter=False`) during Phase 2/3 test procedures.
6. **Window generation**: For records > 5000 samples, exactly 5 distinct, evenly-spaced start positions are deterministically calculated.
7. **MAX aggregation implementation**: Softmax is applied to model logits to extract the class 1 (CVD) probability for each window. The maximum probability among the 5 windows is chosen as the final record-level prediction.
8. **Patient-level split implementation**: 70/15/15 split using random seed 42. Grouping is strictly performed using Patient IDs extracted from file paths (`.../P00001/P00001_E01.hea` -> `P00001`). 
9. **Prediction threshold**: Fixed at `0.5`. (`agg_pred = 1 if agg_prob >= 0.5 else 0`).
10. **Exact target label definition**: `0` if the diagnosis comment contains "normal ecg" (case-insensitive), otherwise `1` (CVD).

## Pipeline Status
The pipeline is formally verified to match the Phase 3 parameters. The model and testing protocols are frozen.
