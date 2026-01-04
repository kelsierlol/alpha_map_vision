import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_fd001(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # FD001 format: unit, time, sensor1..sensor21, setting1..3, RUL label (ignored)
    data = np.loadtxt(path)
    # Keep sensors (cols 2..22) and drop settings/labels
    x = data[:, 2:23].astype(np.float32)
    t = data[:, 1].astype(np.float32)
    unit = data[:, 0].astype(np.int64)
    return unit, t, x


def normalize_train(x_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


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


def stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def main():
    parser = argparse.ArgumentParser(description="CMAPSS Gate1 real split (no injected corruption)")
    parser.add_argument("--data-path", type=str, default="data/train_FD001.txt")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="outputs/gate1_cmapss_real_split.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    unit, t, x = load_fd001(args.data_path)
    # Sort by unit then time to preserve chronology
    order = np.lexsort((t, unit))
    unit, t, x = unit[order], t[order], x[order]

    n = len(x)
    n_train = int(args.train_frac * n)
    x_train = x[:n_train]
    x_eval = x[n_train:]

    x_train_n = normalize_train(x_train, x_train)
    x_eval_n = normalize_train(x_train, x_eval)

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
        s_eval = ((model(torch.from_numpy(x_eval_n).to(device)) - torch.from_numpy(x_eval_n).to(device)).pow(2).mean(dim=1)).cpu().numpy()

    if args.target_fpr and args.target_fpr > 0:
        thresh = float(np.quantile(s_train, 1.0 - args.target_fpr))
    else:
        thresh = float(np.quantile(s_train, 0.99))

    report = {
        "config": vars(args),
        "device": str(device),
        "counts": {"train": int(len(s_train)), "eval": int(len(s_eval))},
        "threshold": {"target_fpr": args.target_fpr, "value": thresh},
        "scores": {"train": stats(s_train), "eval": stats(s_eval)},
        "flag_rate": {
            "train": float((s_train >= thresh).mean()),
            "eval": float((s_eval >= thresh).mean()),
        },
        "status": {
            "eval": "BLOCK" if (s_eval >= thresh).mean() >= 0.25 else ("WARN" if (s_eval >= thresh).mean() >= 0.1 else "PASS"),
            "warn_threshold": 0.10,
            "block_threshold": 0.25,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
