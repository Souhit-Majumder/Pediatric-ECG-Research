# Week 5 — Preprocessing and Advanced CNN Architecture Evaluation

## Objective

Improve upon the Week 4 baseline CNN by systematically evaluating:

1. ECG preprocessing
2. A deeper residual architecture
3. Whether these changes improve generalization
4. Whether the resulting model's improvements are clinically and statistically meaningful

The goal is **not to force an improvement in ROC-AUC**, but to determine whether the proposed changes provide a reproducible improvement over the baseline.

---

## Phase 1 — Establish the Week 4 Baseline

Before changing anything, **freeze the Week 4 baseline**.

Record:
* architecture
* preprocessing
* input shape
* train/test split
* patient-level splitting strategy
* class weights
* random seed
* optimizer
* learning rate
* batch size
* epochs
* ROC-AUC
* accuracy
* sensitivity/recall
* specificity
* precision
* F1
* confusion matrix

This becomes the reference experiment.

### Important
Do **not** change the split while comparing the models.
Otherwise, an apparent improvement could simply come from an easier test set.

---

# Phase 2 — ECG Preprocessing

### Proposed experiment
Introduce ECG filtering **before the signal enters the neural network**.

Potential pipeline:
```text
Raw ECG
   ↓
High-pass filtering
   ↓
Low-pass filtering
   ↓
Optional normalization
   ↓
Fixed-length input
   ↓
ResNet-1D
```

### But don't assume filtering is automatically beneficial
The experiment should answer:
> Does preprocessing improve performance compared with raw ECG input?

Therefore run an explicit comparison:

| Experiment | Input        |
| ---------- | ------------ |
| A          | Raw ECG      |
| B          | Filtered ECG |

Keep everything else identical.

---

## Filtering considerations
Do not blindly choose cutoff frequencies.
The filter parameters should be justified based on:
* ECG frequency content
* sampling rate = 500 Hz
* preservation of clinically relevant morphology
* avoidance of distortion of P/QRS/T components

Prefer a **zero-phase filter** where appropriate so that the filtering process does not introduce a phase shift that changes waveform morphology.

Document:
* filter type
* order
* low cutoff
* high cutoff
* implementation
* rationale

### Very important
Visualize:
```text
Raw ECG
vs
Filtered ECG
```
for representative records.
If filtering visibly distorts P waves, QRS complexes, T waves, or ST segments, then the preprocessing strategy needs to be reconsidered.

---

# Phase 3 — ResNet-1D

Replace the shallow baseline CNN with a 1D residual architecture.

Conceptually:
```text
Input ECG
    ↓
Initial convolution
    ↓
Residual Block
    ↓
Residual Block
    ↓
Residual Block
    ↓
Global pooling
    ↓
Fully connected layer
    ↓
CVD probability
```

## Why ResNet?
As networks become deeper, optimization can become difficult because gradients may become unstable and degradation can occur. Residual connections provide shortcut paths that make optimization of deeper networks easier.

---

# Phase 4 — Architecture Design
Start conservatively. Do NOT immediately build an enormous 20–50-layer model.

For example:
```text
Input
 ↓
Conv1D
 ↓
Residual Block × 2
 ↓
Downsampling
 ↓
Residual Block × 2
 ↓
Downsampling
 ↓
Residual Block × 2
 ↓
Global Average Pooling
 ↓
Linear
 ↓
Output
```

Document:
* kernel sizes
* number of channels
* stride
* pooling/downsampling
* activation
* normalization
* dropout, if used
* number of parameters

The model should be appropriate for the dataset size rather than simply being as deep as possible.

---

# Phase 5 — Controlled Experiments

Run controlled experiments:

### Experiment 1
`Raw ECG + Week 4 CNN` (Baseline)

### Experiment 2
`Filtered ECG + Week 4 CNN` (Measures preprocessing effect)

### Experiment 3
`Raw ECG + ResNet-1D` (Measures architecture effect)

### Experiment 4
`Filtered ECG + ResNet-1D` (Measures combined effect)

---

# Phase 6 — Evaluation
Keep the existing evaluation pipeline **as much as possible**. Evaluate:
* ROC-AUC
* accuracy
* sensitivity
* specificity
* precision
* F1
* confusion matrix
* ROC curve
* Precision-Recall AUC

---

# Phase 7 — Training Curves
Add Figure 1 (Training vs validation loss) and Figure 2 (Training vs validation ROC-AUC). Look for divergence as an indication of overfitting.

---

# Phase 8 — Reproducibility
Use fixed:
* random seed
* patient-level split
* preprocessing parameters
* model configuration

---

# Phase 9 — Statistical Interpretation
Do not write: "AUC increased from 0.55 to 0.68, therefore ResNet works."
Instead: "The ResNet-1D model achieved a higher ROC-AUC than the baseline under the same evaluation protocol."
Then ask: Is the difference substantial? Is it reproducible across seeds? Does preprocessing actually help?

---

# Phase 10 — Avoid Data Leakage
Any preprocessing operation that **learns parameters from the data** must be fitted using training data only. Filtering itself generally does not "learn" parameters from the dataset, but normalization/statistical preprocessing can.

---

# Phase 11 — Clinical/Scientific Interpretation
At the end, explicitly distinguish:
**Model performance**: How accurately does the model discriminate between the defined labels?
from:
**Clinical validity**: Does the model's prediction correspond to clinically meaningful cardiovascular pathology?

---

# Phase 12 — What We Are Actually Trying to Learn
The research question should become:
> **Does appropriate ECG preprocessing and a residual 1D architecture improve pediatric CVD classification compared with the Week 4 baseline under an identical evaluation protocol?**
