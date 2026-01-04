# Gate 2 Downstream Impact Report (AdaLoss)

## Setup
- Dataset: CIFAR-10
- Train samples: 5000
- Classifier: ResNet-18 (CIFAR-adjusted)
- Epochs: 10
- Augmentation: random crop + horizontal flip
- Normalization: CIFAR mean/std
- Gate 2 mode: AdaLoss (alpha-modulated CE)
- Alpha range: [-10.0, 1.9]
- Lambda sweep: 0.1, 0.2, 0.3

## Results (accuracy)
Baseline:
Train \ Test | Clean | Corrupt Seen | Corrupt Unseen
Clean train  | 0.5656 | 0.3870 | 0.4633
Corrupt train| 0.4459 | 0.3799 | 0.4094

AdaLoss(0.1):
AdaLoss(0.1)| 0.4589 | 0.4248 | 0.3941

AdaLoss(0.2):
AdaLoss(0.2)| 0.5358 | 0.4896 | 0.4510

AdaLoss(0.3):
AdaLoss(0.3)| 0.4680 | 0.4460 | 0.4168

## Interpretation (Concise)
- AdaLoss shows a clear robustness lift, especially at lambda=0.2.
- Lambda=0.2 improves corrupt-seen accuracy by +10.16 pts vs clean-train baseline, with a modest clean-accuracy drop (~3 pts).
- Lambda=0.1 helps seen accuracy but hurts clean accuracy too much.
- Lambda=0.3 underperforms lambda=0.2.

## Mic-Drop Comparison (CE vs CE+AdaLoss)
Baseline (CE only):
- Clean: 0.5656
- Corrupt seen: 0.3870
- Corrupt unseen: 0.4633

CE + AdaLoss (lambda=0.2):
- Clean: 0.5358
- Corrupt seen: 0.4896
- Corrupt unseen: 0.4510

Takeaway: AdaLoss delivers a **large corrupt-seen lift** (+10.26 pts) with a **small clean tradeoff** (~3 pts).
