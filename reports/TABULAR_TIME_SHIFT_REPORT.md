# Tabular Time-Slice Shift Report (CreditCard)

## Goal
Demonstrate a product-style **preflight QA gate**:
- Train an unsupervised residual baseline on an earlier “known-good” time slice.
- Score later time slices and flag rows that exceed a high quantile threshold from training.

Dataset: `exp_tabular_fraud/data/creditcard.csv` (normal transactions only for training).

Implementation: `exp_tabular_fraud/tabular_time_shift_demo.py`

## Setup
- Model: MLP autoencoder (MSE)
- Train data: normal transactions (`Class=0`) from earliest time slice (<= train_quantile)
- Threshold: train residual p99
- Scoring: per-row residual energy (mean squared residual)
- max_rows: 200k
- epochs: 3

## Results
Threshold: train p99 = 0.0941

### Natural time drift (no injected schema change)
- Flag rate (train): 1.0% (by definition)
- Flag rate (mid): 12.7%
- Flag rate (late): 14.6%

### Simulated schema change (scale `Amount` in late slice by 5×)
- Flag rate (train): 1.0%
- Flag rate (mid): 12.7%
- Flag rate (late): 31.7%

## Interpretation (Concise)
- Even without injected changes, later time slices deviate from the early slice (preflight warning signal).
- With an upstream “schema change” style perturbation, SafeLoop flags a much larger fraction of the late batch (preflight block signal).

Artifacts:
- JSON: `outputs/tabular_time_shift_report.json` (not committed; generated locally)
