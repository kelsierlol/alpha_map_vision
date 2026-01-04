# CMAPSS (FD001) Report

## Gate 1 (Dropout Detection)
Config: seq_len=50, stride=10, drop_prob=0.7, drop_len=8, drop_value=-3.0  
Output: `outputs/gate1_cmapss_metrics.json`

Metrics:
- PR-AUC: 0.9999
- IoU@k: 1.0000
- FPR@90: 0.0000

## Gate 1 (Real Split, No Injected Corruption)
Script: `timeseries/exp_sensor_dropouts/gate1_cmapss_real_split.py`  
Data: `/Users/prajwal/Projects/supervised_rl/data/train_FD001.txt`  
Split: first 70% (train) / last 30% (eval), calibrated to target FPR=5%

Metrics:
- Threshold (5% FPR): 0.04236
- Flag rate train: 5.01%
- Flag rate eval: 4.67%
- Status: PASS (no drift detected on this split)

## Gate 2 (RUL Impact, Official Split)
Backbone: TCNRegressor (1D conv, hidden=128)  
Corruption: drop_prob=0.7, drop_len=8, drop_value=-3.0

Baseline (TCN):
- clean_rmse: 50.80
- corrupt_rmse: 244.64

SafeLoop (TCN + alpha weighting):
- clean_rmse: 66.58
- corrupt_rmse: 177.24

Interpretation:
- Gate 1 is reliable for simulated dropouts.
- Gate 2 improves robustness (corrupt RMSE) with a clean‑accuracy trade‑off.

## Known Failure Modes
- If the backbone is insensitive to the sequence, corruption won’t change RMSE and Gate 2 is meaningless.
- Over‑weighting via alpha can harm clean RMSE; use “alpha‑light” defaults.
