import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    reference_path: str = "tabular/exp_tabular_fraud/data/hotel_booking_reference_march.csv"
    analysis_path: str = "tabular/exp_tabular_fraud/data/hotel_booking_analysis_march.csv"
    seed: int = 7
    max_rows: int = 0
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    target_fpr: float = 0.05
    output_json: str = "outputs/tabular_hotel_gate1.json"


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


def load_csv_numeric(path: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows and max_rows > 0:
        df = df.iloc[:max_rows]
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] == 0:
        raise ValueError("No numeric columns found; cannot run AE on this dataset.")
    return df_num


def normalize_train(x_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def residual_score(model: TabularAE, x: torch.Tensor) -> torch.Tensor:
    recon = model(x)
    return (recon - x).pow(2).mean(dim=1)


def stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 1 tabular drift demo on hotel booking dataset.")
    parser.add_argument("--reference-path", type=str, default=Config.reference_path)
    parser.add_argument("--analysis-path", type=str, default=Config.analysis_path)
    parser.add_argument("--max-rows", type=int, default=Config.max_rows)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--target-fpr", type=float, default=Config.target_fpr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-json", type=str, default=Config.output_json)
    args = parser.parse_args()

    cfg = Config(
        reference_path=args.reference_path,
        analysis_path=args.analysis_path,
        seed=args.seed,
        max_rows=args.max_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_fpr=args.target_fpr,
        output_json=args.output_json,
    )

    np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ref_df = load_csv_numeric(cfg.reference_path, cfg.max_rows)
    an_df = load_csv_numeric(cfg.analysis_path, cfg.max_rows)
    common = [c for c in ref_df.columns if c in an_df.columns]
    if not common:
        raise ValueError("No shared numeric columns between reference and analysis.")
    ref_df = ref_df[common]
    an_df = an_df[common]
    x_ref = ref_df.to_numpy(dtype=np.float32)
    x_an = an_df.to_numpy(dtype=np.float32)

    x_ref_n = normalize_train(x_ref, x_ref)
    x_an_n = normalize_train(x_ref, x_an)

    model = TabularAE(x_ref_n.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    train_t = torch.from_numpy(x_ref_n).to(device)
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

    model.eval()
    with torch.no_grad():
        s_ref = residual_score(model, torch.from_numpy(x_ref_n).to(device)).cpu().numpy()
        s_an = residual_score(model, torch.from_numpy(x_an_n).to(device)).cpu().numpy()

    if cfg.target_fpr and cfg.target_fpr > 0:
        thresh = float(np.quantile(s_ref, 1.0 - cfg.target_fpr))
    else:
        thresh = float(np.quantile(s_ref, 0.99))

    report = {
        "config": asdict(cfg),
        "device": str(device),
        "counts": {"reference": int(len(s_ref)), "analysis": int(len(s_an))},
        "threshold": {"target_fpr": cfg.target_fpr, "value": thresh},
        "scores": {"reference": stats(s_ref), "analysis": stats(s_an)},
        "flag_rate": {
            "reference": float((s_ref >= thresh).mean()),
            "analysis": float((s_an >= thresh).mean()),
        },
    }

    os.makedirs(os.path.dirname(cfg.output_json), exist_ok=True)
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {cfg.output_json}")


if __name__ == "__main__":
    main()
