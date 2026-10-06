# Week 3: Pediatric ECG Feature Awareness

## 1. Objective
The purpose of this analysis is to computationally extract specific clinically relevant features (Heart Rate, RR interval, QRS duration) from two illustrative pediatric ECG records and explain their medical significance in the context of pediatric cardiology. **These features are descriptive and this is not a diagnostic model.**

## 2. Records Analyzed
- **CVD-labelled Record (`P00001_E01`)**: 1.5 years old (572 days), Female, diagnosed with Ventricular septal defect.
- **Non-CVD-labelled Record (`P04120_E01`)**: ~11 years old (4072 days), Female, Normal ECG.

*Note: These are illustrative examples. Differences cannot be generalized to the broader populations, and pediatric interpretation is highly age-dependent.*

## 3. Feature Extraction Method
- **ECG Preprocessing**: Lead II was selected and cleaned using NeuroKit2's standard ECG cleaning pipeline.
- **R-peak Detection**: R-peaks were detected using NeuroKit2, and RR intervals calculated from peak-to-peak differences.
- **Heart Rate**: Derived mathematically as `60 / Mean RR`.
- **QRS Duration**: QRS onsets and offsets were delineated using the discrete wavelet transform (`dwt`) method in NeuroKit2. Invalid delineations were excluded.

## 4. Extracted Feature Table

| Record     | Age   | Label   | Mean HR (bpm) | Mean RR (s) | Mean QRS (ms) | Valid Beats (QRS/Total) |
| ---------- | ----- | ------- | ------------: | ----------: | ------------: | ----------------------: |
| P00001_E01 | 1.5 y | CVD     | 109.6 | 0.547 | 139.1 | 7/32 |
| P04120_E01 | ~11 y | Non-CVD | 97.6 | 0.615 | 122.6 | 7/32 |

## 5. Heart Rate — Biomedical Significance
Pediatric resting heart rate is strongly age-dependent. Infants and toddlers normally have a significantly higher resting heart rate compared to older children and adults. 
The observed HR of ~110 bpm for the 1.5-year-old and ~98 bpm for the 11-year-old are expected physiological variations based on their ages, and should not be judged using adult reference ranges.

## 6. RR Interval — Biomedical Significance
The RR interval represents the time between successive ventricular depolarizations. It is inversely related to heart rate (shorter RR interval = higher HR). 
The 1.5-year-old has a correspondingly shorter mean RR interval (~0.55 s) than the 11-year-old (~0.61 s).

## 7. QRS Duration — Biomedical Significance
The QRS duration represents the approximate duration of ventricular depolarization. Prolonged QRS duration can be associated with intraventricular conduction abnormalities, conduction delay, or ventricular enlargement/hypertrophy in some contexts.
**However**, it is critical to note that QRS duration alone cannot diagnose VSD, ventricular hypertrophy, or CVD. Pediatric normal ranges for QRS duration also increase slightly with age as the heart grows.

## 8. Comparison of the Two Illustrative Records
The CVD record (`P00001_E01`) has a higher heart rate and shorter RR interval, which is primarily explained by the patient's much younger age (1.5 years vs 11 years). 
We cannot present these simple feature differences as proof of disease discrimination or claim causation.

## 9. Verification / Quality Checks
- Automated checks passed: RR > 0, HR > 0, QRS > 0.
- `Mean HR = 60 / Mean RR` mathematically consistent.
- Validated via visual overlay of detected R-peaks and QRS boundaries.

## 10. Limitations
These descriptive features are not diagnostic biomarkers by themselves. Do not use adult ECG normal ranges as pediatric reference values. The two records are not a statistically representative cohort.
