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

## Results (clean-FPR calibrated, schema change on)
Threshold (target FPR=5%): 0.0191

Flag rates:
- Train (clean): 5.0%
- Mid: 36.8%
- Late (schema-shifted): 97.1%

Fraud proxy (Class=1 rows; unsupervised sanity check):
- Flag rate: 89.6%

## Interpretation (Concise)
- Clean-FPR calibration behaves as expected (≈5% on training slice).
- Mid and late slices show strong drift relative to early slice.
- With a simulated schema shift, late slice is almost entirely flagged → clear “block” signal.

Artifacts:
- JSON: `outputs/tabular_time_shift_report.json` (not committed; generated locally)
