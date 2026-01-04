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
- Features: numeric + categorical (frequency encoding)
- Threshold: calibrated to clean FPR target (5%)
- Epochs: 5

## Results (clean-FPR calibrated)
Threshold (target FPR=5%): 0.1144

Flag rates:
- Reference (clean): 5.0%
- Analysis: 11.9%

Score stats:
- Reference mean 0.0441, p95 0.1144, p99 0.2988
- Analysis mean 0.0692, p95 0.1955, p99 0.3868

## Interpretation (Concise)
- Calibration holds on the reference set (5%).
- Analysis period shows a clear drift signal (flag rate 11.9% vs 5% baseline).
- Frequency encoding preserves categorical signal without exploding residuals.

Notes:
- One-hot encoding + winsor can over-smooth drift and reduce signal.
- No winsorization causes extreme residual blow-ups on this dataset.

## Batch Health (Example Output)
SafeLoop Batch Report
---------------------
Batch: hotel_booking_analysis_march.csv  
Status: WARN  
Flagged: 11.9% (threshold: 10%)  
Recommendation: Review flagged rows before retraining.

Artifacts:
- JSON: `outputs/tabular_hotel_gate1.json` (not committed; generated locally)
