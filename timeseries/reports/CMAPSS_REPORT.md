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

## Gate 1 (Train clean 70%, corrupt last 30% with held-out type)
Script: `timeseries/exp_sensor_dropouts/gate1_cmapss_holdout_corrupt.py`  
Data: `/Users/prajwal/Projects/supervised_rl/data/train_FD001.txt`  
Split: first 70% clean train, last 30% eval with injected corruption

Drift corruption:
- Threshold (train FPR=5%): train 0.04236, detect 0.19278
- Flag rate: train 5.01%, eval 5.01%
- PR-AUC: 0.9998 (mask vs score)

Stuck-at corruption:
- Threshold (train FPR=5%): train 0.04236, detect 0.18872
- Flag rate: train 5.01%, eval 5.01%
- PR-AUC: 0.9998

Interpretation:
- On this split, the detection threshold stays at the clean-calibrated FPR, and the high PR-AUC reflects that the injected corruption is easily separable without raising the flag rate. For a more realistic signal, drift/stuck parameters may need to be weakened or drift measured at per-feature granularity.

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
