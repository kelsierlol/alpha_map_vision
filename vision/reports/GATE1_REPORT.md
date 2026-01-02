# Gate 1 Evaluation Report (Two-Phase Alpha)

## Setup
- Dataset: CIFAR-10
- Train corruptions: occlusion + blur
- Eval corruptions (unseen): saltpepper + copy
- Residual target: clean
- Train samples: 2000
- Epochs: UNet=3, Alpha=3
- Eval batches: 20
- Runs: 3 seeds (7, 8, 9)

## Summary (mean ± std)
Seen corruptions:
- PR-AUC: 0.8626 ± 0.0025
- IoU@k%: 0.6171 ± 0.0018
- IoU@k% (dilated): 0.6691 ± 0.0021
- Hit@k% (dilated): 0.8988 ± 0.0017
- FPR@90% recall: 0.3691 ± 0.0272

Unseen corruptions:
- PR-AUC: 0.8710 ± 0.0034
- IoU@k%: 0.6342 ± 0.0038
- IoU@k% (dilated): 0.6819 ± 0.0023
- Hit@k% (dilated): 0.9616 ± 0.0018
- FPR@90% recall: 0.1131 ± 0.0049

## Per-seed Results
| Seed | Seen PR-AUC | Seen IoU@k% | Seen IoU@k% (dil) | Seen Hit@k% (dil) | Seen FPR@90 | Unseen PR-AUC | Unseen IoU@k% | Unseen IoU@k% (dil) | Unseen Hit@k% (dil) | Unseen FPR@90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.8591 | 0.6170 | 0.6687 | 0.8987 | 0.4070 | 0.8742 | 0.6385 | 0.6787 | 0.9596 | 0.1071 |
| 8 | 0.8641 | 0.6150 | 0.6667 | 0.8968 | 0.3554 | 0.8726 | 0.6348 | 0.6844 | 0.9640 | 0.1131 |
| 9 | 0.8646 | 0.6194 | 0.6718 | 0.9009 | 0.3448 | 0.8663 | 0.6293 | 0.6825 | 0.9612 | 0.1191 |

## Artifacts
- JSON logs: `outputs/eval_runs/run_*.json`

## Interpretation (Concise)
- Gate 1 is stable across seeds with tight variance.
- Performance generalizes to unseen corruptions with **no collapse** in PR-AUC or IoU.
- Unseen FPR@90 is low (≈0.11), indicating practical gating potential on new corruption types.
