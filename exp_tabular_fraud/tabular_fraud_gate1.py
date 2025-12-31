import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    data_path: str = "exp_tabular_fraud/data/creditcard.csv"
    seed: int = 7
    max_rows: int = 200000
    train_frac: float = 0.8
    val_frac: float = 0.1
    batch_size: int = 512
    epochs_ae: int = 5
    epochs_alpha: int = 5
    lr: float = 1e-3
    corrupt_frac: float = 0.25
    missing_frac: float = 0.08
    outlier_frac: float = 0.03
    duplicate_frac: float = 0.05
    output_json: str = "outputs/tabular_fraud_gate1.json"


class TabularAE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class AlphaHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        return self.net(resid)


def load_creditcard_csv(path: str, max_rows: int) -> np.ndarray:
    def conv(b):
        s = b.decode("utf-8").strip().strip('"')
        return float(s)

    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        max_rows=max_rows,
        converters={i: conv for i in range(0, 31)},
    )
    # drop label column (Class) and keep features only
    x = data[:, :-1]
    return x.astype(np.float32)


def split_indices(n: int, train_frac: float, val_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return train, val, test


def normalize_train(x_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def inject_corruption(
    x: torch.Tensor,
    rng: torch.Generator,
    corrupt_frac: float,
    missing_frac: float,
    outlier_frac: float,
    duplicate_frac: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, dim = x.shape
    x_corr = x.clone()
    mask = torch.zeros((bsz, dim), device=x.device)

    # Choose which rows to corrupt
    n_corr = max(1, int(round(bsz * corrupt_frac)))
    rows = torch.randperm(bsz, generator=rng)[:n_corr]

    for r in rows.tolist():
        # missing entries
        n_miss = max(1, int(round(dim * missing_frac)))
        miss_idx = torch.randperm(dim, generator=rng)[:n_miss]
        x_corr[r, miss_idx] = 0.0
        mask[r, miss_idx] = 1.0

        # outliers (add large noise on a few dims)
        n_out = max(1, int(round(dim * outlier_frac)))
        out_idx = torch.randperm(dim, generator=rng)[:n_out]
        scale = 6.0
        x_corr[r, out_idx] = x_corr[r, out_idx] + scale * torch.randn_like(x_corr[r, out_idx])
        mask[r, out_idx] = 1.0

    # duplicates (row-level redundancy)
    n_dup = max(1, int(round(bsz * duplicate_frac)))
    if bsz >= 2 and n_dup > 0:
        src = torch.randperm(bsz, generator=rng)[:n_dup]
        dst = torch.randperm(bsz, generator=rng)[:n_dup]
        x_corr[dst] = x_corr[src]
        mask[dst] = 1.0

    return x_corr, mask


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(labels_sorted.sum(), 1e-12)
    return np.trapz(precision, recall)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeLoop Gate 1 on tabular fraud (synthetic corruption masks).")
    parser.add_argument("--data-path", type=str, default=Config.data_path)
    parser.add_argument("--max-rows", type=int, default=Config.max_rows)
    parser.add_argument("--epochs-ae", type=int, default=Config.epochs_ae)
    parser.add_argument("--epochs-alpha", type=int, default=Config.epochs_alpha)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-json", type=str, default=Config.output_json)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    x = load_creditcard_csv(args.data_path, args.max_rows)

    all_runs = []
    for run in range(args.runs):
        seed = args.seed + run
        cfg = Config(
            data_path=args.data_path,
            seed=seed,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
            epochs_ae=args.epochs_ae,
            epochs_alpha=args.epochs_alpha,
            lr=args.lr,
            output_json=args.output_json,
        )
        rng_np = np.random.default_rng(seed)
        torch.manual_seed(seed)

        train_idx, val_idx, test_idx = split_indices(len(x), cfg.train_frac, cfg.val_frac, rng_np)
        x_train = x[train_idx]
        x_val = x[val_idx]
        x_test = x[test_idx]

        x_train_n = normalize_train(x_train, x_train)
        x_val_n = normalize_train(x_train, x_val)
        x_test_n = normalize_train(x_train, x_test)

        dim = x_train_n.shape[1]
        ae = TabularAE(dim).to(device)
        alpha_head = AlphaHead(dim).to(device)

        opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr)
        ae.train()
        train_t = torch.from_numpy(x_train_n).to(device)
        for epoch in range(1, cfg.epochs_ae + 1):
            perm = torch.randperm(train_t.size(0), device=device)
            total = 0.0
            for i in range(0, train_t.size(0), cfg.batch_size):
                batch = train_t[perm[i:i + cfg.batch_size]]
                recon = ae(batch)
                loss = F.mse_loss(recon, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"[seed {seed}] ae epoch {epoch:02d} | loss {total / max(1, (train_t.size(0) // cfg.batch_size)):.4f}")

        for p in ae.parameters():
            p.requires_grad = False
        ae.eval()
        opt_a = torch.optim.Adam(alpha_head.parameters(), lr=cfg.lr)
        alpha_head.train()
        rng = torch.Generator(device="cpu").manual_seed(seed + 1)

        for epoch in range(1, cfg.epochs_alpha + 1):
            perm = torch.randperm(train_t.size(0), device=device)
            total = 0.0
            for i in range(0, train_t.size(0), cfg.batch_size):
                batch = train_t[perm[i:i + cfg.batch_size]]
                x_corr, mask = inject_corruption(
                    batch.detach(), rng, cfg.corrupt_frac, cfg.missing_frac, cfg.outlier_frac, cfg.duplicate_frac
                )
                with torch.no_grad():
                    recon = ae(x_corr)
                resid = (recon - batch).pow(2).detach()
                logits = alpha_head(resid)
                pos = mask.sum()
                neg = mask.numel() - pos
                pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0).to(device)
                loss = F.binary_cross_entropy_with_logits(logits, mask.to(device), pos_weight=pos_weight)
                opt_a.zero_grad(set_to_none=True)
                loss.backward()
                opt_a.step()
                total += loss.item()
            print(f"[seed {seed}] alpha epoch {epoch:02d} | loss {total / max(1, (train_t.size(0) // cfg.batch_size)):.4f}")

        def eval_split(arr: np.ndarray) -> dict:
            alpha_head.eval()
            t = torch.from_numpy(arr).to(device)
            scores = []
            labels = []
            with torch.no_grad():
                for i in range(0, t.size(0), cfg.batch_size):
                    batch = t[i:i + cfg.batch_size]
                    x_corr, mask = inject_corruption(
                        batch, rng, cfg.corrupt_frac, cfg.missing_frac, cfg.outlier_frac, cfg.duplicate_frac
                    )
                    recon = ae(x_corr)
                    resid = (recon - batch).pow(2)
                    logits = alpha_head(resid)
                    score = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                    scores.append(score)
                    labels.append(mask.cpu().numpy().reshape(-1))
            scores_np = np.concatenate(scores)
            labels_np = np.concatenate(labels)
            return {
                "pr_auc": float(pr_auc(scores_np, labels_np)),
                "iou_k": float(iou_at_k(scores_np, labels_np)),
                "fpr_90": float(fpr_at_recall(scores_np, labels_np, 0.9)),
            }

        metrics_val = eval_split(x_val_n)
        metrics_test = eval_split(x_test_n)
        all_runs.append({"seed": seed, "val": metrics_val, "test": metrics_test})
        print(f"[seed {seed}] val {metrics_val}")
        print(f"[seed {seed}] test {metrics_test}")

    report = {
        "config": asdict(Config(data_path=args.data_path, seed=args.seed, max_rows=args.max_rows, batch_size=args.batch_size, epochs_ae=args.epochs_ae, epochs_alpha=args.epochs_alpha, lr=args.lr, output_json=args.output_json)),
        "device": str(device),
        "runs": all_runs,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
