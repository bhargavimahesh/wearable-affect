# Model card: personalised wrist-based stress detector

## Model details

- **Type**: logistic regression (median imputation → standardisation → L2-regularised logistic regression, `class_weight="balanced"`), scikit-learn 1.9.1
- **Input**: one 60-second window of wrist signals (BVP 64 Hz, EDA 4 Hz, skin temperature 4 Hz, 3-axis acceleration 32 Hz) plus the user's personal baseline
- **Features**: 14 handcrafted features (heart rate and HRV, skin conductance level and responses, skin-temperature level and trend, movement), each expressed as the difference from the user's calm baseline
- **Output**: probability of stress, and a decision at a fixed threshold of 0.5
- **Personal baseline**: mean of each feature over 1–15 minutes of calm sitting (5 minutes recommended), computed via the `/calibrate` endpoint
- **Training data**: all 15 WESAD participants, 880 windows (the first 5 minutes of each recording are used only as calibration)
- **Version**: 0.1.0

## Intended use

- **Intended**: research, teaching, and demonstrating an end-to-end, personalised wearable-ML pipeline.
- **Not intended**: diagnosis, treatment, clinical or occupational decisions, monitoring people without their informed consent, or any use where a wrong prediction could harm someone. This is not a medical device.

## Training and evaluation data

WESAD (Schmidt et al., 2018): 15 participants in a single laboratory session with baseline, amusement, stress (Trier Social Stress Test) and meditation conditions, recorded with an Empatica E4 wristband. Labels are the study **condition**, not a measurement of each person's experience. A manipulation check on the participants' self-reports (SAM arousal) confirmed that all 15 reported higher arousal after the stress task than after baseline.

Task: stress (TSST) vs non-stress (baseline + amusement), on 60 s windows with a 30 s step; only windows lying entirely within one condition; 1,015 windows, 30% stress.

## Evaluation

Leave-one-subject-out: each subject is predicted by a model trained on the other 14. The first 5 minutes of each subject serve as calibration and are not scored.

| | Macro F1 | AUROC |
|---|---|---|
| Generic logistic regression (same windows) | 0.839 | 0.966 |
| **This model (personalised)** | **0.896** | 0.971 |

- Per subject: F1 improved for 9, worsened for 5, unchanged for 1.
- Calibration (pooled LOSO predictions): Brier score 0.076 (base-rate reference 0.225); mean predicted probability 0.359 vs actual stress rate 0.343; slightly overconfident at the top (predicted ~0.93, observed ~0.82).
- This configuration was chosen among 12 evaluated variants on the same subjects, so these numbers are somewhat optimistic. No independent test set exists.

## Limitations and risks

- **Small, homogeneous sample**: 15 participants from one study; performance for other populations, ages and skin tones is unknown.
- **One lab session**: real baselines drift across hours and days; the model has only been evaluated with same-session calibration.
- **Calibration quality matters**: a baseline recorded while the user is not calm, or right after putting on the device (while the EDA signal settles), shifts all later predictions. Personalisation made results worse for 5 of 15 subjects, plausibly for this reason.
- **Individual differences in physiology**: some people show little or even reversed electrodermal response to stress; for them, detection relies on cardiac features.
- **Aroused non-stress states**: amusement and other positive arousal can receive moderate stress probabilities.
- **Signal quality**: wrist PPG is noisy during movement. Heart features that can't be computed are reported as missing, but noisy signals can still yield plausible-looking but wrong values.
- **Lab stressor, one device**: transfer to everyday stress or to other wearables is untested.

## Ethical considerations

Stress is personal and sensitive information. Inferences about it can be wrong, and even correct ones can be misused (e.g. by employers or insurers). Use only with informed consent, keep the user in control of their data, and don't present outputs as diagnoses. The API is stateless by design: it stores no signals, baselines or predictions.