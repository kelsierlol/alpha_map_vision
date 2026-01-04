# Tabular Time-Slice Shift Report (CreditCard)

## Goal
Demonstrate a product-style **preflight QA gate**:
- Train an unsupervised residual baseline on an earlier “known-good” time slice.
- Score later time slices and flag rows that exceed a high quantile threshold from training.

Dataset: `tabular/exp_tabular_fraud/data/creditcard.csv` (normal transactions only for training).

Implementation: `tabular/exp_tabular_fraud/tabular_time_shift_demo.py`

## Setup
- Model: MLP autoencoder (MSE)
- Train data: normal transactions (`Class=0`) from earliest time slice (<= train_quantile)
- Threshold: calibrated to clean FPR (target 5%)
- Scoring: per-row residual energy (mean squared residual)
- max_rows: 200k
- epochs: 5

## Results (clean-FPR calibrated)
Threshold (target FPR=5%): 0.0191

### Baseline (no schema change)
Flag rates:
- Train (clean): 5.0%
- Holdout (clean split-B): 5.6%
- Mid: 36.8%
- Late: 93.4%

Late score stats:
- mean 0.0520, p95 0.1069, p99 0.1952

### Schema change (Amount ×5 in late slice)
Flag rates:
- Train (clean): 5.0%
- Mid: 36.8%
- Late (schema-shifted): 97.1%

Late score stats:
- mean 0.4105, p95 0.7429, p99 5.7882

Fraud proxy (Class=1 rows; unsupervised sanity check):
- Flag rate: 89.6%

## Interpretation (Concise)
- Clean-FPR calibration behaves as expected (≈5% on training slice).
- Split-A/B stability holds (holdout clean slice stays near 5%).
- Even without schema change, later time slices drift away from early slice (preflight warning).
- With schema change, late slice scores jump sharply (p99 explodes), giving a clear “block” signal.

Artifacts:
- JSON: `outputs/tabular_time_shift_report.json` (not committed; generated locally)
