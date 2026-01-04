import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vision.scripts.alpha_map_cifar import TinyUNet2D, corrupt_batch, set_seed
from vision.scripts.alpha_map_cifar_twophase import (
    AlphaHead2D,
    fpr_at_recall,
    hit_topk,
    iou_at_k,
    iou_topk,
    mask_dilate,
    pr_auc,
)


def eval_loop(
    model: TinyUNet2D,
    alpha_head: AlphaHead2D,
    loader: DataLoader,
    rng: torch.Generator,
    eval_modes: Tuple[str, ...],
    calib_thresh: float | None,
    target_mode: str = "clean",
    eval_batches: int = 20,
) -> tuple[dict, list]:
    device = next(model.parameters()).device
    scores = []
    labels = []
    iou_raw = 0.0
    iou_dil = 0.0
    hit_dil = 0.0
    count = 0
    flagged = 0
    total = 0
    overlays = []

    for i, (x, _) in enumerate(loader):
        if i >= eval_batches:
            break
        x = x.to(device)
        corrupt, mask = corrupt_batch(x, rng, modes=eval_modes)
        with torch.no_grad():
            recon = model(corrupt)
            target = x if target_mode == "clean" else corrupt
            resid = (recon - target).pow(2).mean(dim=1, keepdim=True)
            logits = alpha_head(resid)
            score_map = torch.sigmoid(logits)

        score = score_map.cpu().numpy().reshape(-1)
        scores.append(score)
        labels.append(mask.cpu().numpy().reshape(-1))

        if calib_thresh is not None:
            flagged += (score >= calib_thresh).sum()
            total += score.size

        dil_mask = mask_dilate(mask)
        coverage = float(mask.mean().item())
        iou_raw += iou_topk(score_map, mask, coverage)
        iou_dil += iou_topk(score_map, dil_mask, coverage)
        hit_dil += hit_topk(score_map, dil_mask, coverage)
        count += 1

        if len(overlays) < 3:
            overlays.append(
                {
                    "corrupt": corrupt[0].cpu(),
                    "alpha": score_map[0, 0].cpu(),
                    "mask": mask[0, 0].cpu(),
                }
            )

    scores_np = np.concatenate(scores) if scores else np.array([])
    labels_np = np.concatenate(labels) if labels else np.array([])

    metrics = {
        "pr_auc": pr_auc(scores_np, labels_np) if scores else 0.0,
        "iou_k": iou_at_k(scores_np, labels_np) if scores else 0.0,
        "fpr_90": fpr_at_recall(scores_np, labels_np, 0.9) if scores else 0.0,
        "flagged": float(flagged / total) if calib_thresh is not None and total else None,
    }
    if count:
        metrics["iou_k_dilated"] = iou_dil / count
        metrics["hit_k_dilated"] = hit_dil / count
    return metrics, overlays


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot transfer: CIFAR-trained alpha → STL10 with synthetic corruptions.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--weights-dir", type=str, default="weights_twophase")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-modes", type=str, default="occlusion,blur,saltpepper,copy", help="comma list")
    parser.add_argument("--residual-target", type=str, default="clean", choices=["clean", "input"])
    parser.add_argument("--target-fpr", type=float, default=0.0, help="Calibrate threshold on clean STL10.")
    parser.add_argument("--calib-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="outputs/vision_transfer/stl10_transfer.json")
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_modes = tuple(m.strip() for m in args.eval_modes.split(",") if m.strip())

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    stl_test = datasets.STL10(root=args.data_dir, split="test", download=True, transform=transform)
    test_loader = DataLoader(stl_test, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TinyUNet2D().to(device)
    alpha_head = AlphaHead2D(in_ch=1).to(device)
    model_path = os.path.join(args.weights_dir, "unet.pth")
    alpha_path = os.path.join(args.weights_dir, "alpha_head.pth")
    if not os.path.exists(model_path) or not os.path.exists(alpha_path):
        raise FileNotFoundError(f"Missing weights in {args.weights_dir}; expected unet.pth and alpha_head.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    alpha_head.load_state_dict(torch.load(alpha_path, map_location=device))
    model.eval()
    alpha_head.eval()

    calib_thresh = None
    if args.target_fpr and args.target_fpr > 0:
        clean_scores = []
        for i, (x, _) in enumerate(test_loader):
            if i >= args.calib_batches:
                break
            x = x.to(device)
            with torch.no_grad():
                recon = model(x)
                target = x if args.residual_target == "clean" else x
                resid = (recon - target).pow(2).mean(dim=1, keepdim=True)
                logits = alpha_head(resid)
                score_map = torch.sigmoid(logits)
            clean_scores.append(score_map.detach().cpu().numpy().reshape(-1))
        if clean_scores:
            clean_scores_np = np.concatenate(clean_scores)
            calib_thresh = np.quantile(clean_scores_np, 1.0 - args.target_fpr)
            clean_fpr = float((clean_scores_np >= calib_thresh).mean())
            print(f"Calibrated threshold @ target FPR={args.target_fpr:.3f}: {calib_thresh:.4f} (clean FPR {clean_fpr:.4f})")

    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    metrics, overlays = eval_loop(
        model,
        alpha_head,
        test_loader,
        rng,
        eval_modes,
        calib_thresh,
        target_mode=args.residual_target,
        eval_batches=args.eval_batches,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report = {"config": vars(args), "metrics": metrics}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {args.output}")
    print(
        "Transfer metrics on STL10 (corruption score = 1 - alpha): "
        f"PR-AUC {metrics['pr_auc']:.4f}, IoU@k {metrics['iou_k']:.4f}, FPR@90 {metrics['fpr_90']:.4f}"
    )

    if args.save_overlays and overlays:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = os.path.dirname(args.output)
        for idx, ov in enumerate(overlays):
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(ov["corrupt"].permute(1, 2, 0).numpy())
            axs[0].axis("off")
            axs[0].set_title("Corrupted")
            axs[1].imshow(ov["alpha"].numpy(), cmap="viridis")
            axs[1].axis("off")
            axs[1].set_title("Alpha map")
            axs[2].imshow(ov["mask"].numpy(), cmap="Reds")
            axs[2].axis("off")
            axs[2].set_title("Corruption mask")
            fig.tight_layout()
            outfile = os.path.join(out_dir, f"stl10_overlay_{idx}.png")
            fig.savefig(outfile, dpi=150)
            plt.close(fig)
            print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
