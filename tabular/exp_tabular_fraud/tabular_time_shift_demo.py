import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    data_path: str = "exp_tabular_fraud/data/creditcard.csv"
    seed: int = 7
    max_rows: int = 200000
    train_quantile: float = 0.6
    holdout_quantile: float = 0.7
    mid_quantile: float = 0.8
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    threshold_quantile: float = 0.99
    target_fpr: float = 0.05
    simulate_schema_change: bool = True
    schema_shift_scale: float = 5.0
    report_fraud_proxy: bool = True
    output_json: str = "outputs/tabular_time_shift_report.json"


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


def load_creditcard(path: str, max_rows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = []
    feats = []
    labels = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            row = [c.strip().strip('"') for c in row]
            # Time, V1..V28, Amount, Class
            times.append(float(row[0]))
            feats.append([float(x) for x in row[0:-1]])  # include Time in features for schema-shift simulation
            labels.append(int(float(row[-1])))
    t = np.asarray(times, dtype=np.float32)
    x = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return t, x, y


def normalize_train(x_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def residual_score(model: TabularAE, x: torch.Tensor) -> torch.Tensor:
    recon = model(x)
    resid = (recon - x).pow(2).mean(dim=1)
    return resid


def stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeLoop preflight demo: detect time-sliced shift (creditcard.csv).")
    parser.add_argument("--data-path", type=str, default=Config.data_path)
    parser.add_argument("--max-rows", type=int, default=Config.max_rows)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-quantile", type=float, default=Config.train_quantile)
    parser.add_argument("--holdout-quantile", type=float, default=Config.holdout_quantile)
    parser.add_argument("--mid-quantile", type=float, default=Config.mid_quantile)
    parser.add_argument("--threshold-quantile", type=float, default=Config.threshold_quantile)
    parser.add_argument("--target-fpr", type=float, default=Config.target_fpr)
    parser.add_argument("--simulate-schema-change", action="store_true", default=Config.simulate_schema_change)
    parser.add_argument("--no-simulate-schema-change", action="store_false", dest="simulate_schema_change")
    parser.add_argument("--schema-shift-scale", type=float, default=Config.schema_shift_scale)
    parser.add_argument("--report-fraud-proxy", action="store_true", default=Config.report_fraud_proxy)
    parser.add_argument("--no-report-fraud-proxy", action="store_false", dest="report_fraud_proxy")
    parser.add_argument("--output-json", type=str, default=Config.output_json)
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data_path,
        seed=args.seed,
        max_rows=args.max_rows,
        train_quantile=args.train_quantile,
        holdout_quantile=args.holdout_quantile,
        mid_quantile=args.mid_quantile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold_quantile=args.threshold_quantile,
        target_fpr=args.target_fpr,
        simulate_schema_change=args.simulate_schema_change,
        schema_shift_scale=args.schema_shift_scale,
        report_fraud_proxy=args.report_fraud_proxy,
        output_json=args.output_json,
    )

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    t, x, y = load_creditcard(cfg.data_path, cfg.max_rows)

    # Keep only normal transactions for training the residual baseline.
    normal = (y == 0)
    t_n = t[normal]
    x_n = x[normal]

    # Define time cutoffs
    t0 = np.quantile(t_n, cfg.train_quantile)
    t_holdout = np.quantile(t_n, cfg.holdout_quantile)
    t1 = np.quantile(t_n, cfg.mid_quantile)

    train_mask = t_n <= t0
    holdout_mask = (t_n > t0) & (t_n <= t_holdout)
    mid_mask = (t_n > t_holdout) & (t_n <= t1)
    late_mask = t_n > t1

    x_train = x_n[train_mask]
    x_holdout = x_n[holdout_mask]
    x_mid = x_n[mid_mask]
    x_late = x_n[late_mask]

    # "Schema change" simulation: scale Amount (last feature before Class; in our x includes Time..Amount)
    # Here we scale the Amount column in the late split to emulate upstream unit/scale bug.
    x_late_eval = x_late.copy()
    if cfg.simulate_schema_change:
        amount_col = x_late_eval.shape[1] - 1  # Amount is last column in x (Class excluded)
        x_late_eval[:, amount_col] = x_late_eval[:, amount_col] * cfg.schema_shift_scale

    # Normalize using train split
    x_train_n = normalize_train(x_train, x_train)
    x_holdout_n = normalize_train(x_train, x_holdout)
    x_mid_n = normalize_train(x_train, x_mid)
    x_late_n = normalize_train(x_train, x_late_eval)

    dim = x_train_n.shape[1]
    model = TabularAE(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    train_t = torch.from_numpy(x_train_n).to(device)
    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(train_t.size(0), device=device)
        total = 0.0
        for i in range(0, train_t.size(0), cfg.batch_size):
            batch = train_t[perm[i:i + cfg.batch_size]]
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"ae epoch {epoch:02d} | loss {total / max(1, (train_t.size(0) // cfg.batch_size)):.4f}")

    # Score splits
    model.eval()
    with torch.no_grad():
        s_train = residual_score(model, torch.from_numpy(x_train_n).to(device)).cpu().numpy()
        s_holdout = residual_score(model, torch.from_numpy(x_holdout_n).to(device)).cpu().numpy()
        s_mid = residual_score(model, torch.from_numpy(x_mid_n).to(device)).cpu().numpy()
        s_late = residual_score(model, torch.from_numpy(x_late_n).to(device)).cpu().numpy()

    if cfg.target_fpr and cfg.target_fpr > 0:
        thresh = float(np.quantile(s_train, 1.0 - cfg.target_fpr))
    else:
        thresh = float(np.quantile(s_train, cfg.threshold_quantile))
    report = {
        "config": asdict(cfg),
        "device": str(device),
        "counts": {"train": int(len(s_train)), "holdout": int(len(s_holdout)), "mid": int(len(s_mid)), "late": int(len(s_late))},
        "threshold": {"quantile": cfg.threshold_quantile, "target_fpr": cfg.target_fpr, "value": thresh},
        "scores": {
            "train": stats(s_train),
            "holdout": stats(s_holdout),
            "mid": stats(s_mid),
            "late": stats(s_late),
        },
        "flag_rate": {
            "train": float((s_train >= thresh).mean()),
            "holdout": float((s_holdout >= thresh).mean()),
            "mid": float((s_mid >= thresh).mean()),
            "late": float((s_late >= thresh).mean()),
        },
    }

    if cfg.report_fraud_proxy:
        fraud = (y == 1)
        if fraud.any():
            x_f = normalize_train(x_train, x[fraud])
            with torch.no_grad():
                s_fraud = residual_score(model, torch.from_numpy(x_f).to(device)).cpu().numpy()
            report["fraud_proxy"] = {
                "count": int(len(s_fraud)),
                "scores": stats(s_fraud),
                "flag_rate": float((s_fraud >= thresh).mean()),
            }

    os.makedirs(os.path.dirname(cfg.output_json), exist_ok=True)
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {cfg.output_json}")


if __name__ == "__main__":
    main()
