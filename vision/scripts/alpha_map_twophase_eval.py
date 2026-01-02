import argparse
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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


def eval_metrics(
    model: TinyUNet2D,
    alpha_head: AlphaHead2D,
    loader: DataLoader,
    rng: torch.Generator,
    device: torch.device,
    modes: Tuple[str, ...],
    residual_target: str,
    eval_batches: int,
) -> Dict[str, float]:
    scores = []
    labels = []
    iou_raw = 0.0
    iou_dil = 0.0
    hit_dil = 0.0
    count = 0

    for i, (x, _) in enumerate(loader):
        if i >= eval_batches:
            break
        x = x.to(device)
        corrupt, mask = corrupt_batch(x, rng, modes=modes)
        with torch.no_grad():
            recon = model(corrupt)
            target = x if residual_target == "clean" else corrupt
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
    return {
        "pr_auc": float(pr_auc(scores_np, labels_np)),
        "iou_k": float(iou_at_k(scores_np, labels_np)),
        "iou_k_dil": float(iou_dil / max(count, 1)),
        "hit_k_dil": float(hit_dil / max(count, 1)),
        "fpr_90": float(fpr_at_recall(scores_np, labels_np, 0.9)),
    }


def train_unet(
    model: TinyUNet2D,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"unet epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")


def train_alpha(
    model: TinyUNet2D,
    alpha_head: AlphaHead2D,
    loader: DataLoader,
    rng: torch.Generator,
    device: torch.device,
    lr: float,
    epochs: int,
    train_modes: Tuple[str, ...],
    residual_target: str,
) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    opt = torch.optim.Adam(alpha_head.parameters(), lr=lr)
    alpha_head.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            corrupt, mask = corrupt_batch(x, rng, modes=train_modes)
            with torch.no_grad():
                recon = model(corrupt)
            target = x if residual_target == "clean" else corrupt
            resid = (recon - target).pow(2).mean(dim=1, keepdim=True).detach()

            logits = alpha_head(resid)
            pos = mask.sum()
            neg = mask.numel() - pos
            pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"alpha epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase evaluation with unseen corruptions.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-unet", type=int, default=3)
    parser.add_argument("--epochs-alpha", type=int, default=3)
    parser.add_argument("--lr-unet", type=float, default=1e-3)
    parser.add_argument("--lr-alpha", type=float, default=5e-4)
    parser.add_argument("--train-modes", type=str, default="occlusion,blur")
    parser.add_argument("--eval-modes", type=str, default="saltpepper,copy")
    parser.add_argument("--residual-target", type=str, default="clean", choices=["clean", "input"])
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=str, default="outputs/eval_runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_modes = tuple(m.strip() for m in args.train_modes.split(",") if m.strip())
    eval_modes = tuple(m.strip() for m in args.eval_modes.split(",") if m.strip())

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    results = []
    for run in range(args.runs):
        seed = args.seed + run
        set_seed(seed)
        rng = torch.Generator(device=device).manual_seed(seed + 1)

        idx = torch.randperm(len(train_set))[: args.max_samples]
        subset = Subset(train_set, idx.tolist())
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=False)

        model = TinyUNet2D().to(device)
        alpha_head = AlphaHead2D(in_ch=1).to(device)

        print(f"\n=== Run {run + 1}/{args.runs} | seed {seed} ===")
        train_unet(model, train_loader, device, args.lr_unet, args.epochs_unet)
        train_alpha(
            model,
            alpha_head,
            train_loader,
            rng,
            device,
            args.lr_alpha,
            args.epochs_alpha,
            train_modes,
            args.residual_target,
        )

        model.eval()
        alpha_head.eval()
        seen = eval_metrics(
            model,
            alpha_head,
            test_loader,
            rng,
            device,
            train_modes,
            args.residual_target,
            args.eval_batches,
        )
        unseen = eval_metrics(
            model,
            alpha_head,
            test_loader,
            rng,
            device,
            eval_modes,
            args.residual_target,
            args.eval_batches,
        )

        run_result = {
            "seed": seed,
            "seen": seen,
            "unseen": unseen,
        }
        results.append(run_result)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.save_dir, f"run_{run + 1}_{stamp}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": vars(args),
                    "result": run_result,
                },
                f,
                indent=2,
            )
        print(f"Saved {out}")

    def agg(metric: str, key: str) -> Tuple[float, float]:
        vals = [r[key][metric] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n=== Summary (mean ± std) ===")
    for key in ("seen", "unseen"):
        print(f"{key}:")
        for metric in ("pr_auc", "iou_k", "iou_k_dil", "hit_k_dil", "fpr_90"):
            mean, std = agg(metric, key)
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
