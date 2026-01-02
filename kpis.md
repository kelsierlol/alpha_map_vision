# SafeLoop KPIs (Production-Readiness Targets)

This file defines the minimum metrics and evaluation scope for a product-grade SafeLoop release.  
The goal is to validate **one time per modality** (vision / time-series / tabular), then ship a
frozen default + light calibration for new datasets (no per-dataset tuning).

## Vision (Images)
**Primary metrics**
- PR-AUC vs corruption mask (pixel-level).
- IoU@k (top-k alpha vs mask coverage).
- FPR@90% recall (or TPR@5% FPR if stricter).
- Coverage % (avg fraction flagged per image).
- Qualitative: top-N overlays per run.

**Datasets**
- CIFAR-10/100 (fast baseline).
- STL-10 (transfer test, no retraining).
- Stretch: ImageNet subset.

**Production-ready KPIs**
- Seen corruptions: PR-AUC ≥ 0.80, IoU@k ≥ 0.40, FPR@90 ≤ 0.25.
- Unseen corruptions: PR-AUC ≥ 0.60.
- Seed stability: std < 0.02 across 3 seeds.
- Cross-dataset transfer (CIFAR → STL) maintains PR-AUC ≥ 0.60.

**Current roadblocks**
- Legacy scripts use `recon - corrupt` residuals; not the canonical two-phase signal.
- Some corruption types (copy/blur) can be subtle and degrade localization.
- Repair/weighting can harm clean accuracy when coverage is too high.

## Time-Series (Sensors)
**Primary metrics**
- PR-AUC / IoU@k / FPR@90 vs injected corruption masks.
- Coverage % per window.
- Drift sensitivity: clean vs corrupted batch trust score shift.
- Top-N worst windows (qualitative).

**Datasets**
- CMAPSS FD001 (dropout, drift, stuck-at, mixed).
- Stretch: SWaT or NAB subset.

**Production-ready KPIs**
- Mixed-corruption training: PR-AUC ≥ 0.70 on mixed test.
- Unseen corruption type: PR-AUC ≥ 0.50.
- FPR@90 ≤ 0.30.
- Drift: corrupted batch scores clearly higher than clean baseline.

**Current roadblocks**
- Some backbones are insensitive to corruption (must validate sensitivity).
- Training on a single corruption type does not generalize to others.

## Tabular
**Primary metrics (row-level by default)**
- Row-level PR-AUC for corrupted/outlier rows.
- FPR@90 at row-level.
- Batch drift score vs reference (PSI/KL or model-based).
- Coverage % and top-N flagged rows.

**Datasets**
- Credit Card Fraud (time-shift + row-level trust).
- Adult/Census or Telco Churn (mixed types, missingness).

**Production-ready KPIs**
- Row-level PR-AUC ≥ 0.70 on held-out corruption/drift.
- FPR@90 ≤ 0.25.
- Drift: late batch flagged rate increases when a known shift is injected.
- Stable across 3 seeds (std < 0.02).

**Current roadblocks**
- Production-style `input_target` residuals are weak for feature-level localization.
- Duplicates/redundancy are not reliably detected by residuals alone (needs similarity module).
- Feature-level mask claims should be limited; row-level trust + drift is more robust.

## Gate Decision KPIs (Cross-cutting)
- PASS/WARN/BLOCK thresholds calibrated to a target clean FPR (e.g., 5%).
- 3-seed reporting for all headline metrics.
- One cross-dataset transfer test per modality (no retraining).
- Runtime: complete in minutes on a single GPU/CPU for demo datasets.

## Current Scope Summary
- **Vision Gate 1** is closest to production-ready (strong localization + visuals).
- **Gate 2 (repair/weighting)** is promising but needs an oracle check and coverage sweep to avoid
  hurting clean accuracy.
- **Tabular** should be positioned as row-level trust + drift, not feature-level masks.
