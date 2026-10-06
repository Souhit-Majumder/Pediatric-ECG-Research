# Week 1 — Professor Discussion Guide
## Explainable AI–Based Detection of Pediatric Cardiovascular Diseases Using Multi-Lead ECG Signals

---

## A. What I Completed

During Week 1, I focused on **rigorous dataset familiarization** without prematurely training any machine-learning models.

I:

- Set up the Python environment and required ECG-processing libraries, including `wfdb`.
- Audited the structure of the ZZU pediatric ECG dataset.
- Loaded and examined:
  - `AttributesDictionary.csv`
  - `DiseaseCode.csv`
  - `ECGCode.csv`
- Investigated how the metadata correspond to the `.hea` and `.dat` WFDB waveform files.
- Examined:
  - patient counts
  - ECG-record counts
  - age distribution
  - gender distribution
  - lead configurations
  - sampling frequency
  - recording duration
  - disease-label structure
- Investigated the distinction between **patient-level information and ECG-record-level information**.
- Investigated the 9-lead and 12-lead ECG configurations.
- Loaded actual ECG waveform signals from the WFDB records.
- Generated single-lead and multi-lead ECG visualizations.
- Practiced identifying the basic:
  - P wave
  - QRS complex
  - T wave
- Examined the distinction between **clinical disease labels** and **ECG diagnostic statements**.

### Main objective of Week 1

The objective was not to obtain model performance.

The objective was to answer:

> **What exactly is contained in this dataset, how is each ECG represented, and what does an individual pediatric ECG record look like?**

---

# B. What I Learned Technically

## 1. WFDB

The ZZU pECG dataset stores ECG recordings using the **Waveform Database (WFDB) format**.

WFDB provides a structured way of storing physiological waveform records together with their associated metadata.

---

## 2. `.hea` and `.dat` Files

Each WFDB recording consists primarily of:

### `.hea`

The header contains information such as:

- sampling frequency
- number of signals
- signal/lead information
- signal metadata

### `.dat`

The `.dat` file contains the actual digitized ECG waveform data.

Therefore, the header and signal file must be interpreted together when loading an ECG.

---

## 3. Metadata CSV Files

The dataset provides additional information through CSV files.

These allow the waveform records to be associated with information such as:

- demographic information
- disease codes
- ECG diagnostic codes
- other signal-related metadata

An important lesson was that these different codes should **not automatically be treated as equivalent labels**.

---

## 4. Patient vs ECG Record

The dataset contains:

- **14,190 ECG records**
- **11,643 children**

Therefore:

> **One ECG record does not necessarily represent one unique patient.**

Some children have multiple ECG recordings.

### Why this matters

Later, when we train a machine-learning model, randomly splitting ECG records could potentially place recordings from the same child into both training and testing sets.

This could lead to **patient-level data leakage**.

Therefore, I believe future model development should use **patient-level splitting/grouping**.

---

## 5. Lead Configuration

The dataset contains both:

- 12-lead ECG records
- 9-lead ECG records

The dataset paper reports:

| Configuration | ECG records |
|---|---:|
| 12-lead | 12,334 |
| 9-lead | 1,856 |

The 9-lead configuration does not contain:

- V2
- V4
- V6

The dataset paper explains that obtaining all six precordial leads can be difficult in younger children.

### Important implication

I should **not simply treat V2/V4/V6 as ordinary missing numerical values and blindly impute them**.

The absence of these leads is a structural characteristic of part of the dataset.

---

## 6. Sampling Frequency

The dataset uses a sampling frequency of:

> **500 Hz**

This means that the ECG signal is sampled 500 times per second.

This will become important when converting samples into time and when designing the eventual CNN input representation.

---

## 7. Recording Duration

The ECG recordings have variable durations.

The dataset paper reports that most recordings fall within approximately the **20–40 second range**.

This creates an important future modeling question:

> How should variable-length ECG recordings be converted into a consistent CNN input?

Possible approaches include:

- fixed-length truncation
- padding
- fixed-length segmentation

I will **not make this modeling decision during Week 1**.

---

# C. What I Learned Biologically

## 1. 12-Lead ECG

A standard 12-lead ECG provides multiple electrical views of cardiac activity.

These leads can broadly be divided into:

### Limb leads

- I
- II
- III

### Augmented limb leads

- aVR
- aVL
- aVF

### Precordial/chest leads

- V1
- V2
- V3
- V4
- V5
- V6

The limb and augmented leads primarily provide information from the frontal plane, while the precordial leads provide information from the horizontal plane.

---

## 2. P Wave

The **P wave** represents atrial depolarization.

---

## 3. QRS Complex

The **QRS complex** represents ventricular depolarization.

It is typically the largest and most visually prominent component of a normal ECG cycle.

---

## 4. T Wave

The **T wave** represents ventricular repolarization.

---

## 5. Pediatric ECG

Pediatric ECG interpretation differs from adult ECG interpretation because cardiac electrical characteristics and normal ECG morphology change throughout childhood.

Therefore:

> **Age is not simply a demographic variable; it is potentially relevant to ECG morphology and model behavior.**

This supports the later Week 6 age-wise analysis defined in the project deliverables.

---

## 6. 9-Lead Pediatric ECGs

The dataset paper explains that the 9-lead configuration occurs because obtaining all six precordial leads can be difficult in younger children.

Therefore, the absence of V2, V4, and V6 should be understood as a **dataset acquisition characteristic**, rather than automatically being treated as corrupted or incomplete data.

---

# D. Important Dataset Observations

## 1. Age Distribution

The age distribution is not uniform across the 0–14-year range.

### Research implication

The model may encounter substantially different amounts of data across pediatric age ranges.

Therefore, age distribution needs to be considered when interpreting future model performance.

---

## 2. Disease-Label Imbalance

The cardiovascular disease categories are not evenly represented.

Some disease categories have substantially more records than others.

### Research implication

Later model evaluation should not rely solely on accuracy.

Metrics such as:

- sensitivity
- specificity
- precision
- F1-score
- ROC-AUC

may become important depending on the final classification task.

---

## 3. Patient-Level vs Record-Level Data

There are more ECG records than unique children.

This means the dataset has a longitudinal/repeated-record component.

### Research implication

Patient-level separation should be considered during model development to avoid leakage.

---

## 4. 9-Lead vs 12-Lead Records

Approximately:

> **1,856 / 14,190 ≈ 13.1%**

of ECG records use the 9-lead configuration.

This is a substantial subset and should not simply be discarded without justification.

---

## 5. Clinical Disease vs ECG Interpretation

One of the most important conceptual observations is that:

> **A clinical disease label and an ECG diagnostic statement are not necessarily the same thing.**

A disease diagnosis comes from the clinical record, whereas the ECG diagnostic information describes the interpretation of the ECG itself.

Therefore, a disease-labelled patient may not necessarily exhibit an obviously abnormal waveform in the particular ECG recording.

### Research implication

This distinction will be particularly important during:

- Week 2 — signal comparison
- Week 4 — classification
- Week 7 — failure analysis

---

# E. Questions I Recommend Discussing With My Professor

## Priority 1 — Initial Classification Target

> **Which cardiovascular disease category should we select for the first binary CNN experiment in Week 4?**

This should be decided after examining class sizes and label quality.

---

## Priority 2 — Lead Strategy

> **For the initial CNN, should we use a common lead subset, or should we explicitly design the model to account for both 9-lead and 12-lead ECGs?**

This is important because simply dropping all 9-lead records would discard approximately 13% of the ECG records.

---

## Priority 3 — Multi-Label Records

> **How should records associated with multiple cardiovascular disease labels be handled in the initial binary classification experiment?**

We need to decide whether the first experiment should use:

- a carefully defined binary subset
- exclusion criteria
- multi-label formulation later

---

## Priority 4 — Patient-Level Validation

> **Should we use patient-grouped cross-validation in addition to a patient-level train/validation/test split?**

This would help prevent repeated recordings from the same child from appearing across different partitions.

---

## Priority 5 — Age Groups

> **Which pediatric age grouping should we use for the Week 6 age-wise analysis?**

The grouping should ideally have a clinical rationale rather than being selected only because it produces convenient sample sizes.

---

## Priority 6 — Disease Label vs ECG Appearance

> **If a patient has a clinical cardiovascular disease label but the ECG itself appears visually normal, should that record remain in the classification target, or should these cases be analyzed separately during failure analysis?**

This is an important question because the clinical disease label and the observable ECG morphology are not necessarily equivalent.

---

# F. Potential Research Risks

## 1. Class Imbalance

A model may learn to favor the majority class.

Therefore, high accuracy alone may not indicate good disease detection.

---

## 2. Patient Leakage

Multiple ECG records may belong to the same child.

Random record-level splitting could therefore create leakage.

### Mitigation

Use patient-level grouping when creating the experimental partitions.

---

## 3. Lead Availability

The 9-lead records are valid pediatric ECG recordings, but they do not contain V2, V4, and V6.

Possible future strategies include:

- common-lead modeling
- lead-aware architecture
- separate sensitivity analysis

I would avoid artificial ECG-lead imputation unless there is a strong methodological justification.

---

## 4. Label Ambiguity

A clinical disease diagnosis does not guarantee that the corresponding ECG will contain an obvious abnormality.

Therefore, some apparent model "errors" may require clinical interpretation rather than being immediately dismissed as model failures.

---

## 5. Age Imbalance

Some age groups may have substantially fewer ECG records.

This could affect:

- model training
- performance estimates
- explanation stability

---

## 6. Variable Recording Duration

The ECG recordings have different durations.

Before CNN training, we will need a principled strategy for obtaining consistent model inputs.

---

# G. What Week 2 Will Build Upon

Week 1 answered:

> **What is in the dataset?**

Week 2 will ask:

> **What does the ECG signal actually look like, and how might disease affect its morphology?**

The next step will be to compare ECG recordings from healthy/non-CVD and disease-labelled children.

The comparison will focus on:

- waveform morphology
- P wave
- QRS complex
- T wave
- rhythm
- visually apparent differences
- lead-specific differences

The goal is **biomedical interpretation**, not machine learning.

This follows the project deliverables, which define Week 2 as comparison of a healthy child and a diseased child with written observations.

---

# H. 60–90 Second Explanation to My Professor

> “This week I focused on establishing a rigorous understanding of the ZZU pediatric ECG dataset before moving into machine learning. I audited the metadata and WFDB waveform files and learned how the `.hea` and `.dat` files work together.
>
> One important finding was that there are 14,190 ECG records but only 11,643 children, so some children have multiple recordings. This means that when we eventually train the model, we need to be careful about patient-level data splitting to avoid leakage.
>
> I also investigated the lead configurations. The dataset contains both 12-lead and 9-lead ECGs, with 1,856 records using the 9-lead configuration. The dataset paper explains that obtaining all six precordial leads can be difficult in younger children, so I don't think we should simply treat the missing V2, V4, and V6 signals as ordinary missing numerical values.
>
> On the biomedical side, I loaded actual ECG signals and created single-lead and multi-lead visualizations. I also practiced identifying the P wave, QRS complex, and T wave and understanding the roles of the different ECG leads.
>
> Another important observation was that clinical disease labels and ECG diagnostic statements are different concepts. This will be important when we interpret model errors later.
>
> For Week 2, I will now move from dataset understanding to biomedical signal comparison by examining healthy and disease-labelled pediatric ECGs and documenting their morphological differences before we begin deep learning.” 

---

# I. Key Takeaways I Should Remember

If my professor asks:

### “What was the biggest technical finding?”

> **The distinction between patient-level and ECG-record-level data, because it directly affects how we must split the dataset later.**

### “What was the biggest pediatric-specific finding?”

> **The coexistence of 9-lead and 12-lead ECGs, with the 9-lead configuration being a structural characteristic of the dataset.**

### “What was the biggest biomedical finding?”

> **Pediatric ECG morphology changes with development, so age needs to be considered when interpreting ECG patterns and future model behavior.**

### “What was the biggest research concern?”

> **We cannot treat clinical disease labels, ECG appearance, and ECG diagnostic statements as interchangeable concepts.**

### “What is the next step?”

> **Week 2: systematically compare healthy and disease-labelled pediatric ECG morphology before beginning machine learning.**

---

# Week 1 Status

**Dataset familiarization:** ✅  
**WFDB understanding:** ✅  
**Metadata understanding:** ✅  
**Lead configuration analysis:** ✅  
**Patient/record distinction:** ✅  
**Basic ECG visualization:** ✅  
**P-QRS-T identification:** ✅  
**Disease-label investigation:** ✅  
**CNN:** Not started — intentionally  
**XAI:** Not started — intentionally  
**Age-wise model analysis:** Not started — intentionally  

> **Week 1 establishes the data and biomedical foundation required for the subsequent modeling stages.**