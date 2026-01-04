import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_fd001(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    x = data[:, 2:23].astype(np.float32)
    t = data[:, 1].astype(np.float32)
    unit = data[:, 0].astype(np.int64)
    return unit, t, x


def normalize_train(x_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def corrupt_batch(x: torch.Tensor, mode: str = "drift", rng: torch.Generator | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, dim = x.shape
    x_corr = x.clone()
    mask = torch.zeros_like(x_corr)
    if mode == "drift":
        scale = torch.rand((bsz, dim), device=x.device, generator=rng) * 0.5 + 1.5  # 1.5..2.0
        x_corr = x_corr * scale
        mask = torch.ones_like(x_corr)
    elif mode == "stuck":
        frac = 0.2
        n_feats = int(dim * frac)
        idx = torch.randperm(dim, generator=rng, device=x.device)[:n_feats]
        x_corr[:, idx] = 0.0
        mask[:, idx] = 1.0
    else:
        raise ValueError(f"Unknown mode {mode}")
    return x_corr, mask


class AE1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def main():
    parser = argparse.ArgumentParser(description="CMAPSS Gate1: train clean 70%, corrupt last 30% with holdout corruption")
    parser.add_argument("--data-path", type=str, default="data/train_FD001.txt")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--corrupt-mode", type=str, default="drift", choices=["drift", "stuck"])
    parser.add_argument("--output", type=str, default="outputs/gate1_cmapss_holdout_corrupt.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    rng = torch.Generator(device=device).manual_seed(args.seed + 123)

    unit, t, x = load_fd001(args.data_path)
    order = np.lexsort((t, unit))
    unit, t, x = unit[order], t[order], x[order]

    n = len(x)
    n_train = int(args.train_frac * n)
    x_train = x[:n_train]
    x_eval_clean = x[n_train:]

    x_train_n = normalize_train(x_train, x_train)
    x_eval_n = normalize_train(x_train, x_eval_clean)

    model = AE1D(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    train_t = torch.from_numpy(x_train_n).to(device)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(train_t.size(0), device=device)
        total = 0.0
        for i in range(0, train_t.size(0), args.batch_size):
            batch = train_t[perm[i:i + args.batch_size]]
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"ae epoch {epoch:02d} | loss {total / max(1, train_t.size(0) // args.batch_size):.4f}")

    model.eval()
    with torch.no_grad():
        s_train = ((model(train_t) - train_t).pow(2).mean(dim=1)).cpu().numpy()

    if args.target_fpr and args.target_fpr > 0:
        thresh = float(np.quantile(s_train, 1.0 - args.target_fpr))
    else:
        thresh = float(np.quantile(s_train, 0.99))

    # Corrupt the held-out slice
    x_eval_t = torch.from_numpy(x_eval_n).to(device)
    x_corr, mask = corrupt_batch(x_eval_t, mode=args.corrupt_mode, rng=rng)
    with torch.no_grad():
        s_eval = ((model(x_corr) - x_corr).pow(2).mean(dim=1)).cpu().numpy()
    mask_flat = mask.mean(dim=1).cpu().numpy() > 0  # any feature corrupted → mark row
    labels = mask_flat.astype(np.int64)

    def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        order = np.argsort(-scores)
        labels_sorted = labels[order]
        tp = np.cumsum(labels_sorted)
        fp = np.cumsum(1 - labels_sorted)
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / np.maximum(labels_sorted.sum(), 1e-12)
        trap = getattr(np, "trapezoid", np.trapz)
        return float(trap(precision, recall))

    scores = s_eval
    labels = labels
    pr = pr_auc(scores, labels) if labels.sum() > 0 else 0.0
    thresh_detect = np.quantile(scores, 1.0 - args.target_fpr) if args.target_fpr > 0 else thresh
    flag_rate = float((scores >= thresh_detect).mean())

    report = {
        "config": vars(args),
        "device": str(device),
        "counts": {"train": int(len(s_train)), "eval": int(len(scores))},
        "threshold": {"target_fpr": args.target_fpr, "train_value": thresh, "detect_value": float(thresh_detect)},
        "scores": {"train": stats(s_train), "eval": stats(scores)},
        "flag_rate": {"train": float((s_train >= thresh).mean()), "eval": flag_rate},
        "pr_auc": pr,
        "mask_positives": int(labels.sum()),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
