# Gate 2 Downstream Impact Report (Alpha-Guided Repair)

## Setup
- Dataset: CIFAR-10
- Train samples: 5000
- Classifier: ResNet-18 (CIFAR-adjusted)
- Epochs: 10
- Augmentation: random crop + horizontal flip
- Normalization: CIFAR mean/std
- Repair policy: top-k=5%, blend=0.2, dilate=1

## Results (accuracy)
Train \ Test | Clean | Corrupt Seen | Corrupt Unseen
Clean train  | 0.6025 | 0.4270 | 0.4873
Corrupt train| 0.4209 | 0.3816 | 0.3510
Alpha-repair | 0.4744 | 0.4353 | 0.4109

## Interpretation (Concise)
- Alpha‑repair improves corrupted‑test accuracy vs corrupt‑train baseline:
  - Seen: +5.37 pts
  - Unseen: +5.99 pts
- Clean accuracy drops vs clean‑train, but remains above corrupt‑train.
- Coverage is light (≈5%), consistent with “alpha‑light” strategy.
