# Alpha Map Vision MVP (CIFAR-10)

Small, fast demo to show alpha heatmaps lighting up corrupted patches on real-world images.

## What it does
- Trains a tiny UNet-style autoencoder on CIFAR-10 images.
- Injects known corruptions (occlusion, blur, salt/pepper, copy-paste) with exact masks.
- Trains an alpha controller to output low alpha where corruption exists.
- Exports a 3-panel PNG: corrupted image, alpha map, corruption mask.

## Run
```bash
python alpha_map_cifar.py --plot
```

### Faster / cooler laptop
```bash
python alpha_map_cifar.py --max-samples 1000 --epochs-unet 2 --epochs-alpha 2 --batch-size 64 --plot
```

### Output
PNG saved to `outputs/cifar_alpha_demo.png`

## Evaluation + Generalization Test
Runs PR-AUC, IoU@k%, FPR@90% recall on a validation set, and a generalization test
(train on occlusion+blur, test on salt/pepper+copy). Optionally saves weights.

```bash
python alpha_map_eval.py --max-samples 2000 --epochs-unet 3 --epochs-alpha 3 --eval-batches 20 --save-weights
```

## Two-Phase Variant
See `TWO_PHASE.md` for rationale and usage.
