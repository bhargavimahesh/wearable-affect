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