# Findings

## Data
- All 15 subjects load; signal durations match label durations exactly, confirming the documented
  sampling rates. Recordings last 87–118 min.
- ~67 windows per subject, ~20 of them stress (~10 min); stress fraction 29.8% overall.
- Stress shows higher skin conductance level (3.4 vs 1.3 µS), more SCRs (~20 vs ~7 per min),
  higher heart rate (89 vs 76 bpm) and falling skin temperature.
- EDA levels differ hugely between people: S7's calm level (4.6 µS) is higher than most people's
  stress level. Absolute values mean little without a personal baseline.
- S14 barely responds on EDA (0.29 → 0.34 µS); S17's SCL even drops under stress.

## Baseline models (LOSO, 15 subjects)
| Model | Macro F1 | Balanced acc. | AUROC |
|---|---|---|---|
| Dummy (always non-stress) | 0.41 ± 0.00 | 0.50 ± 0.00 | 0.50 ± 0.00 |
| Logistic regression | 0.83 ± 0.11 | 0.85 ± 0.10 | 0.97 ± 0.05 |
| LightGBM | 0.85 ± 0.14 | 0.87 ± 0.12 | 0.98 ± 0.04 |

- Logistic regression and LightGBM are effectively tied (each better for 7 of 15 subjects), but
  fail on different people (e.g. S14: logreg 0.89, LightGBM 0.42; S13: 0.64 vs 1.00).
- AUROC ≥ 0.86 for every subject and both models: the generic model ranks stress well for everyone.
- Errors come from person-specific probability shifts, not from failing to recognise stress.
  S14 (LightGBM): AUROC 0.91 but F1 0.42 — all probabilities below 0.5 (stress max 0.20).
- Oracle per-subject threshold (upper bound only, uses test labels): mean F1 0.85 → 0.91;
  best thresholds range 0.05–0.95.
- LightGBM's probabilities are extreme and poorly calibrated (many optimal thresholds at the
  0.05 / 0.95 grid edges; S14 calm windows ~0.0001).

## Implications
- Personalisation (normalising to a personal baseline) should help people with shifted levels
  (S7, S13), but cannot create an absent response (S14); there, cardiac features must carry the
  signal — an argument for multimodal models.
- Probability calibration needs attention before the API returns probabilities to users.

## PaPaGei-S (LOSO, LightGBM / logistic regression)
- PPG only: PaPaGei embeddings (F1 0.77–0.78, AUROC 0.87–0.88) roughly match but do not beat
  4 handcrafted heart features (F1 0.78–0.80, AUROC 0.89). Plausible cause: clinical
  pretraining data; stress in WESAD appears mainly as a heart-rate change.
- Full model: logistic regression gets worse with PaPaGei (F1 0.83 -> 0.81, AUROC 0.97 -> 0.92),
  consistent with overfitting to 512 dimensions. LightGBM: F1 up (0.85 -> 0.88) but AUROC down
  (0.98 -> 0.95), so the F1 gain likely reflects threshold effects, not better separation.
  Per-subject analysis pending.
- Per subject (LightGBM, all handcrafted vs non-heart handcrafted + PaPaGei):
  - Mean F1 gain (0.85 -> 0.88) comes almost entirely from two subjects. S10: F1 0.74 -> 0.97
    with AUROC 1.00 in both (pure threshold effect). S14: F1 0.42 -> 0.67 but AUROC 0.91 -> 0.68.
  - AUROC: 3 subjects better, 8 worse, 4 tied at 1.00. Sign test excluding ties: p ≈ 0.23
    (not significant).
  - S14 (EDA non-responder) depends on cardiac information; replacing handcrafted heart features
    with PaPaGei removed the signal the model relied on, and ranking dropped sharply.
- Conclusion: no evidence PaPaGei-S improves on handcrafted heart features on WESAD; F1 gains are
  threshold effects. Likely cause: clinical pretraining (domain shift to wrist PPG).
- Lesson: F1 gains can hide ranking losses; always check AUROC per subject alongside F1.

## Manipulation check using self-reported arousal and valence
- Across subjects, self-reported arousal change was not related to SCL change (Spearman
  ρ = 0.09, p = 0.74) or to model AUROC (ρ = 0.18, p = 0.51). With n = 15, only strong
  correlations (|ρ| >= 0.5) would be detectable, and SAM is a coarse 1–9 scale with ceiling
  effects, so this is absence of evidence, not evidence of absence.

## Personalisation (calibration = first minutes of each recording)
- Success criterion (fixed in advance): mean macro F1 higher than generic at the same
  calibration length, AND more subjects improve than get worse.
- Baseline normalisation (subtract personal calm mean, fixed 0.5 threshold) raised mean F1 in all
  6 model/length combinations; strongest for logistic regression (5 min: 0.839 -> 0.896).
  AUROC barely changed (~0.97): normalisation fixes per-person level shifts, not ranking.
- Per-subject criterion met in 5/6 combinations; 10-min logistic regression (highest mean, 0.908)
  failed it (7 better / 7 worse). Best split 10/4; no single comparison significant with n = 15
  (sign test p ≈ 0.18).
- Predictions made before running held: S7 0.79 -> 0.89 and S10 0.90 -> 1.00 improved; S14
  unchanged (EDA non-responder); S17 slightly worse.
- Worse for S11 (0.96 -> 0.76), S9, S3. Hypothesis: the first minutes after putting on the device
  are not representative (EDA electrode settling, participants still settling in).
- Personal threshold (95th percentile of calibration probabilities) was worse than 0.5 in all
  12 comparisons. Quiet sitting does not represent all non-stress states (e.g. amusement), so the
  threshold is too low and produces false alarms; worst for LightGBM's extreme probabilities.
  Rejected; calm-only calibration is useful for features, not for thresholds.
- Chosen for deployment: logistic regression + baseline normalisation, 5-min calibration, fixed
  0.5 threshold (F1 0.896, 9 better / 5 worse). Selected after seeing results among 12 variants,
  so its score is somewhat optimistic.
