# Week 6 — Preprocessing and Advanced CNN Architecture Evaluation

## Phase 0 — Freeze the current baseline
Preserve the current experiments exactly:
E1 Raw CNN              best Val AUC ≈ 0.6015
E2 Filtered CNN         best Val AUC ≈ 0.5932
E3 Raw ResNet-1D        best Val AUC = 0.8165
E4 Filtered ResNet-1D  best Val AUC = 0.8117
**Do not overwrite these models/results.**

## Phase 1 — Fix the training protocol
Before changing architecture:
* patient-level split
* fixed random seed
* same train/validation/test split for all experiments
* validation-based checkpoint selection
* early stopping
* AdamW
* learning-rate scheduler
* 30–50 maximum epochs
* save best validation-AUC checkpoint
* test set untouched until final model selection
Goal: determine whether the current ResNet has simply been under-trained.

## Phase 2 — Improve temporal sampling
Experiment with random 10-sec training windows instead of the first 5000 samples.

## Phase 3 — Lead handling
Investigate the current zero-padding of 9-lead ECGs. Implement an explicit 12x1 lead-availability mask.

## Phase 4 — ECG augmentation
Test controlled augmentation (Gaussian noise, amplitude scaling, baseline-wander perturbation, temporal masking, conservative lead dropout).

## Phase 5 — Model improvement
Only after the above, test a Multi-scale / InceptionTime-style 1D model.

## Phase 6 — Final evaluation
For the best model, evaluate comprehensive metrics and run Grad-CAM.

## Phase 7 — Scientific interpretation
Determine whether better temporal coverage, lead handling, training, augmentation, and multi-scale representation improve patient-independent pediatric CVD classification.
