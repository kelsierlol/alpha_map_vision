# Tabular Fraud (CreditCard) Report

## Gate 1 (Feature-Level Corruption Detection)
Dataset: `exp_tabular_fraud/data/creditcard.csv` (features only; label column is ignored for Gate 1).  
Approach: MLP autoencoder (MSE) + alpha head trained on **clean-target residuals** with synthetic corruption masks.

Synthetic corruptions (for ground truth masks):
- Missing entries (set to 0)
- Outlier injection (add large noise)
- Duplicates (row-level redundancy proxy)

## 3-Seed Results (200k rows, 3+3 epochs)
Reported metrics are computed on held-out splits with fresh corruption injection:

Validation (mean ± std):
- PR-AUC: 0.8438 ± 0.0008
- IoU@k: 0.6335 ± 0.0042
- FPR@90: 0.1311 ± 0.0016

Test (mean ± std):
- PR-AUC: 0.8462 ± 0.0034
- IoU@k: 0.6354 ± 0.0019
- FPR@90: 0.1287 ± 0.0011

Artifacts:
- Metrics JSON: `outputs/tabular_fraud_gate1.json`
