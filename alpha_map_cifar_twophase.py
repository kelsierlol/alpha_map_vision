import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from alpha_map_cifar import TinyUNet2D, corrupt_batch, local_redundancy, set_seed


class AlphaHead2D(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, resid_mag: torch.Tensor) -> torch.Tensor:
        return self.net(resid_mag)


def norm01(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(labels_sorted.sum(), 1e-12)
    return np.trapezoid(precision, recall)


def iou_at_k(scores: np.ndarray, labels: np.ndarray) -> float:
    coverage = labels.mean()
    if coverage <= 0:
        return 0.0
    thresh = np.quantile(scores, 1.0 - coverage)
    preds = scores >= thresh
    inter = np.logical_and(preds, labels).sum()
    union = np.logical_or(preds, labels).sum()
    return float(inter / max(union, 1))


def fpr_at_recall(scores: np.ndarray, labels: np.ndarray, recall_target: float = 0.9) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    recall = tp / max(labels_sorted.sum(), 1e-12)
    idx = np.searchsorted(recall, recall_target, side="left")
    if idx >= len(fp):
        idx = len(fp) - 1
    return float(fp[idx] / max((1 - labels_sorted).sum(), 1e-12))


def mask_dilate(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)


def mask_erode(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)


def iou_topk(scores: torch.Tensor, mask: torch.Tensor, coverage: float) -> float:
    if coverage <= 0:
        return 0.0
    thresh = torch.quantile(scores.flatten(), 1.0 - coverage)
    preds = (scores >= thresh).float()
    inter = (preds * mask).sum().item()
    union = ((preds + mask) > 0).sum().item()
    return float(inter / max(union, 1))


def hit_topk(scores: torch.Tensor, mask: torch.Tensor, coverage: float) -> float:
    if coverage <= 0:
        return 0.0
    thresh = torch.quantile(scores.flatten(), 1.0 - coverage)
    preds = (scores >= thresh).float()
    hits = (preds * mask).sum().item()
    return float(hits / max(preds.sum().item(), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase alpha map demo on CIFAR-10.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-unet", type=int, default=3)
    parser.add_argument("--epochs-alpha", type=int, default=3)
    parser.add_argument("--lr-unet", type=float, default=1e-3)
    parser.add_argument("--lr-alpha", type=float, default=5e-4)
    parser.add_argument("--alpha-target", type=str, default="mask", choices=["mask", "residual"])
    parser.add_argument("--train-modes", type=str, default="occlusion", help="comma list")
    parser.add_argument("--eval-modes", type=str, default="occlusion,blur,saltpepper,copy", help="comma list")
    parser.add_argument("--residual-target", type=str, default="clean", choices=["clean", "input"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--weights-dir", type=str, default="weights_twophase")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    idx = torch.randperm(len(train_set))[: args.max_samples]
    subset = Subset(train_set, idx.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TinyUNet2D().to(device)
    alpha_head = AlphaHead2D(in_ch=1).to(device)

    # Phase 1: Train UNet with MSE for faithful reconstruction
    opt_unet = torch.optim.Adam(model.parameters(), lr=args.lr_unet)
    model.train()
    for epoch in range(1, args.epochs_unet + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            opt_unet.zero_grad(set_to_none=True)
            loss.backward()
            opt_unet.step()
            total += loss.item()
        print(f"unet epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")

    # Phase 2: Freeze UNet, train alpha head on residuals
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    opt_alpha = torch.optim.Adam(alpha_head.parameters(), lr=args.lr_alpha)
    alpha_head.train()
    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    train_modes = tuple(m.strip() for m in args.train_modes.split(",") if m.strip())
    eval_modes = tuple(m.strip() for m in args.eval_modes.split(",") if m.strip())

    for epoch in range(1, args.epochs_alpha + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            corrupt, mask = corrupt_batch(x, rng, modes=train_modes)
            x_in = corrupt
            with torch.no_grad():
                recon = model(x_in)
            target = x if args.residual_target == "clean" else x_in
            resid = (recon - target).pow(2).mean(dim=1, keepdim=True).detach()

            if args.alpha_target == "mask":
                target_mask = mask
                logits = alpha_head(resid)
                pos = target_mask.sum()
                neg = target_mask.numel() - pos
                pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
                loss = F.binary_cross_entropy_with_logits(logits, target_mask, pos_weight=pos_weight)
            else:
                target = 1.0 - norm01(resid)
                pred = torch.sigmoid(alpha_head(resid))
                loss = F.mse_loss(pred, target)
            opt_alpha.zero_grad(set_to_none=True)
            loss.backward()
            opt_alpha.step()
            total += loss.item()
        print(f"alpha epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")

    if args.save_weights:
        os.makedirs(args.weights_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.weights_dir, "unet.pth"))
        torch.save(alpha_head.state_dict(), os.path.join(args.weights_dir, "alpha_head.pth"))
        print(f"Saved weights to {args.weights_dir}")

    if args.plot:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x, _ = next(iter(loader))
        x = x.to(device)
        corrupt, mask = corrupt_batch(x, rng)
        with torch.no_grad():
            recon = model(corrupt)
            target = x if args.residual_target == "clean" else corrupt
            resid = (recon - target).pow(2).mean(dim=1, keepdim=True)
            logits = alpha_head(resid)
            alpha = torch.sigmoid(logits)

        os.makedirs(args.output_dir, exist_ok=True)
        img = corrupt[0].permute(1, 2, 0).cpu().numpy()
        heat = alpha[0, 0].cpu().numpy()
        m = mask[0, 0].cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img)
        axs[0].set_title("Corrupted")
        axs[0].axis("off")
        axs[1].imshow(heat, cmap="viridis")
        axs[1].set_title("Trust map (alpha)")
        axs[1].axis("off")
        axs[2].imshow(m, cmap="Reds")
        axs[2].set_title("Corruption mask")
        axs[2].axis("off")
        fig.tight_layout()
        out = os.path.join(args.output_dir, "cifar_alpha_twophase.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")

    # Quick eval on test set
    alpha_head.eval()
    scores = []
    labels = []
    iou_raw = 0.0
    iou_dil = 0.0
    hit_dil = 0.0
    count = 0
    for i, (x, _) in enumerate(test_loader):
        if i >= args.eval_batches:
            break
        x = x.to(device)
        corrupt, mask = corrupt_batch(x, rng, modes=eval_modes)
        with torch.no_grad():
            recon = model(corrupt)
            target = x if args.residual_target == "clean" else corrupt
            resid = (recon - target).pow(2).mean(dim=1, keepdim=True)
            logits = alpha_head(resid)
            score_map = torch.sigmoid(logits)
        score = score_map.cpu().numpy().reshape(-1)
        scores.append(score)
        labels.append(mask.cpu().numpy().reshape(-1))

        dil_mask = mask_dilate(mask)
        coverage = float(mask.mean().item())
        iou_raw += iou_topk(score_map, mask, coverage)
        iou_dil += iou_topk(score_map, dil_mask, coverage)
        hit_dil += hit_topk(score_map, dil_mask, coverage)
        count += 1

    scores_np = np.concatenate(scores)
    labels_np = np.concatenate(labels)
    print("Eval (corruption score = 1 - alpha)")
    print(f"PR-AUC: {pr_auc(scores_np, labels_np):.4f}")
    print(f"IoU@k%: {iou_at_k(scores_np, labels_np):.4f}")
    if count:
        print(f"IoU@k% (dilated mask): {iou_dil / count:.4f}")
        print(f"Hit@k% (dilated mask): {hit_dil / count:.4f}")
    print(f"FPR@90% recall: {fpr_at_recall(scores_np, labels_np, 0.9):.4f}")


if __name__ == "__main__":
    main()
