# Week 2: Biomedical Signal Understanding Observations

*Note: The selected ECGs (`P00001_E01` and `P04120_E01`) are illustrative examples used for signal familiarization and qualitative biomedical interpretation. Differences or similarities observed between these individual records cannot be generalized to the broader CVD-labelled and non-CVD-labelled populations.*

## 1. Literature Context (ECG_Paper1.pdf)
The study establishing the ZZU pECG dataset highlights several critical differences between pediatric and adult cardiology. Children undergo significant anatomical and physiological changes during growth, resulting in ECG waveforms (heart rate, intervals, lead amplitude, etc.) that are substantially divergent from adults. The incidence of specific diseases also varies, with congenital heart disease, myocarditis, and Kawasaki disease being more common in pediatrics. Additionally, due to insufficient cooperation or small chest sizes in younger children, extracting all 6 precordial leads is often intractable; hence, a 9-lead configuration (I, II, III, aVR, aVL, aVF, V1, V3, V5) is clinically valid for children under 7. Our selected CVD-labelled record (`P00001_E01`, age ~1.5 years) uses this 9-lead configuration, which is a structural characteristic of the data and not an error.

## 2. Metadata Summary
- **CVD-labelled Record (`P00001_E01`)**: Female, age 572 days (~1.57 years). Diagnosed with Ventricular septal defect (ICD-10: Q21.0). ECG Statement: Left ventricular high voltage. Configuration: 9-lead. Sampling Rate: 500 Hz.
- **Non-CVD-labelled Record (`P04120_E01`)**: Female, age 4072 days (~11 years). No cardiovascular disease codes (ICD-10: J03.9, J34.1, J35.2). ECG Statement: Normal ECG. Configuration: 12-lead. Sampling Rate: 500 Hz.

## 3. Structured Signal Analysis

### Level 1: Lead II Comparison
- **Observation:** Both records demonstrate regular repeating cycles with stable baselines and visually identifiable P, QRS, and T components. The CVD-labelled record has an approximate qualitative heart rate that appears slightly higher (more beats in the 5-second window) than the non-CVD-labelled record.
- **Interpretation:** The regular rhythm suggests normal sinus rhythm in both children. The difference in heart rate may reflect normal physiologic variation, autonomic tone at the time of recording, or age-dependent developmental differences (younger children typically have higher resting heart rates).
- **Limitation:** A slightly higher heart rate in the 572-day-old child cannot be diagnostically attributed to the Ventricular Septal Defect (VSD) simply by visual inspection of this single 5-second excerpt.

### Level 2: Multi-lead Comparison
- **Observation:** In the CVD-labelled record, there is a pronounced amplitude in the QRS complexes in the left-sided leads (e.g., Lead I, aVL, V5) compared to the Non-CVD record. The 9-lead structural limitation is visible as V2, V4, V6 are absent in the CVD-labelled plot. 
- **Interpretation:** The notably larger amplitudes in the CVD-labelled record align qualitatively with the clinical ECG diagnostic statement of "Left ventricular high voltage", which can be associated with ventricular septal defects leading to ventricular hypertrophy. 
- **Limitation:** Voltage criteria for hypertrophy are highly age-dependent in pediatrics due to thinner chest walls and physiological right ventricular dominance shifting leftward over the first few years of life. We cannot definitively state that these high voltages are pathological without applying strict, age-adjusted criteria.

### Level 3: Zoomed Cardiac Cycle
- **Observation:** For both records, the P wave is upright in Lead II preceding the QRS, and the QRS is narrow. The T wave is visible and follows the QRS without abnormal ST-segment elevation or depression. The QRS complex in the CVD-labelled cycle reaches a higher absolute amplitude than the non-CVD cycle in Lead II.
- **Interpretation:** The upright P wave indicates normal sinoatrial node depolarization. The narrow QRS complex indicates normal intraventricular conduction. 
- **Limitation:** Visual identification of these waves is qualitative. Without automated, precise measurements of the PR interval or QRS duration, we cannot draw definitive conclusions about conduction delays or structural anomalies from this single cycle.

## 4. Pediatric Context Synthesis
The most striking visible difference between the two illustrative records is the QRS amplitude. While `P00001_E01`'s high voltage correlates with its VSD label and its specific ECG diagnostic statement, its younger age (572 days) must be factored in. Normal pediatric physiology allows for wide variations in QRS amplitude. This analysis successfully moves our understanding from metadata structure to qualitative waveform familiarity, establishing a biomedical foundation without overstating the diagnostic capabilities of manual visual inspection.
