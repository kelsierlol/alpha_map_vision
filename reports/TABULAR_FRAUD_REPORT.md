# Tabular Fraud (CreditCard) Report

## Gate 1 (Feature-Level Corruption Detection)
Dataset: `exp_tabular_fraud/data/creditcard.csv` (features only; label column is ignored for Gate 1).  
Approach: MLP autoencoder (MSE) + alpha head trained on **clean-target residuals** with synthetic corruption masks.

Why clean-target residuals?
- This is an **evaluation harness**: we inject corruptions with known masks and compare recon(corrupt) vs clean to get a strong supervision signal.
- In production, clean targets are not available; SafeLoop uses input-target residuals plus calibration/aggregation signals.

Synthetic corruptions (for ground truth masks):
- Missing entries (set to an out-of-distribution sentinel in normalized space)
- Outlier injection (replace with large noise)
- Note: duplicates are a redundancy problem and are not included in the corruption-mask metrics (residuals alone don’t reliably detect duplicates).

## 3-Seed Results (200k rows, 3+3 epochs)
Reported metrics are computed on held-out splits with fresh corruption injection.

We report two evaluation modes:
- `clean_target` (evaluation harness): residual is computed vs the original clean row (only possible because we injected the corruption and still have the clean source row in memory).
- `input_target` (production-style): residual is computed vs the observed input row (what you’d have in production).

### `clean_target` (evaluation harness)
Validation (mean ± std):
- PR-AUC: 0.7207 ± 0.0104
- IoU@k: 0.5800 ± 0.0101
- FPR@90: 0.0692 ± 0.0011

Test (mean ± std):
- PR-AUC: 0.7276 ± 0.0041
- IoU@k: 0.5830 ± 0.0077
- FPR@90: 0.0691 ± 0.0026

### `input_target` (production-style)
Validation (mean ± std):
- PR-AUC: 0.2993 ± 0.0164
- IoU@k: 0.1997 ± 0.0096
- FPR@90: 0.2375 ± 0.0024

Test (mean ± std):
- PR-AUC: 0.2971 ± 0.0142
- IoU@k: 0.2019 ± 0.0100
- FPR@90: 0.2419 ± 0.0017

Interpretation:
- The large gap between `clean_target` and `input_target` is expected: without access to clean targets, “reconstruction residual vs input” is a weaker supervision signal for *localizing* synthetic feature corruptions.
- This is why SafeLoop’s production story can’t rely on residual magnitude alone for all tabular issues; it needs calibration + aggregation (row-level scoring, cohort/time drift) and (for some issue types) additional signals (e.g., missingness indicators, redundancy/similarity detectors for duplicates).

Artifacts:
- Metrics JSON: `outputs/tabular_fraud_gate1.json`
- Metrics JSON (`input_target`, 3+3 epochs): `outputs/tabular_fraud_gate1_input_target_3ep.json`
