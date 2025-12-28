# Two-Phase Alpha Mapping

## Why two phases?
SafeLoop’s goal is not anomaly reconstruction. It’s to **score trustworthiness** of data.
If you use a robust loss during reconstruction, the model learns to ignore spikes
and anomalies. That’s great for robustness, but it destroys the signal you need
for a reliable trust score.

So we decouple the objectives:

1) **Phase 1: Faithful reconstruction**  
Train the backbone with MSE so it reconstructs *everything* (including anomalies).
This gives stable, unbiased residuals.

2) **Phase 2: Trust scoring**  
Freeze the backbone. Train a small alpha head to convert residual patterns into a
trust map. The alpha head sees **residuals only** (detached), so it can learn
where data is unreliable without pushing the backbone to ignore signal.

## What you get
- A **trust map** (alpha) that highlights low-confidence regions.
- A reconstruction backbone that stays honest and stable.
- A clean separation between “model fidelity” and “data quality scoring.”

## Scripts
- `scripts/alpha_map_cifar_twophase.py` implements the two-phase pipeline for CIFAR-10.

## When to use
Use two-phase when your product is **preflight QA / promotion gating** and
you want a reliable trust score rather than anomaly detection.
