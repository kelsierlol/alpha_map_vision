# Tabular Gate 1 Report (Hotel Booking)

## Goal
Show a production-style **preflight drift gate** on a real tabular dataset
using a clean reference set and an analysis set.

Dataset:
- Reference: `tabular/exp_tabular_fraud/data/hotel_booking_reference_march.csv`
- Analysis: `tabular/exp_tabular_fraud/data/hotel_booking_analysis_march.csv`

Implementation: `tabular/exp_tabular_fraud/tabular_hotel_gate1.py`

## Setup
- Model: MLP autoencoder (MSE)
- Features: numeric columns only (intersection of ref + analysis)
- Threshold: calibrated to clean FPR target (5%)
- Epochs: 5

## Results (clean-FPR calibrated)
Threshold (target FPR=5%): 0.0632

Flag rates:
- Reference (clean): 5.0%
- Analysis: 53.4%

Score stats:
- Reference mean 0.0283, p95 0.0632, p99 0.1613
- Analysis mean 0.0900, p95 0.1739, p99 0.4610

## Interpretation (Concise)
- Calibration holds on the reference set (5%).
- Analysis period shows a large shift in residual distribution (flag rate 53%).
- This mirrors the “performance drop” window NannyML highlights — SafeLoop would block or warn before retraining.

Artifacts:
- JSON: `outputs/tabular_hotel_gate1.json` (not committed; generated locally)
