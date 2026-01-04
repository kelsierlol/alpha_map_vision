import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

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
    drop_cols: str = "is_canceled,Index,y_pred,y_pred_proba"
    winsor_low: float = 0.01
    winsor_high: float = 0.99
    categorical_encoding: str = "freq"
    apply_winsor: bool = True
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


def load_csv(path: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows and max_rows > 0:
        df = df.iloc[:max_rows]
    return df


def drop_columns(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=cols), cols


def encode_features(ref_df: pd.DataFrame, an_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([ref_df, an_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, drop_first=False)
    ref_enc = combined.iloc[: len(ref_df)].reset_index(drop=True)
    an_enc = combined.iloc[len(ref_df):].reset_index(drop=True)
    return ref_enc, an_enc


def normalize_train(x_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def winsorize_train(x_train: np.ndarray, x: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    low = np.quantile(x_train, low_q, axis=0, keepdims=True)
    high = np.quantile(x_train, high_q, axis=0, keepdims=True)
    return np.clip(x, low, high)


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


def batch_status(flag_rate: float, warn: float = 0.10, block: float = 0.25) -> str:
    if flag_rate >= block:
        return "BLOCK"
    if flag_rate >= warn:
        return "WARN"
    return "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 1 tabular drift demo on hotel booking dataset.")
    parser.add_argument("--reference-path", type=str, default=Config.reference_path)
    parser.add_argument("--analysis-path", type=str, default=Config.analysis_path)
    parser.add_argument("--max-rows", type=int, default=Config.max_rows)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--target-fpr", type=float, default=Config.target_fpr)
    parser.add_argument("--drop-cols", type=str, default=Config.drop_cols)
    parser.add_argument("--winsor-low", type=float, default=Config.winsor_low)
    parser.add_argument("--winsor-high", type=float, default=Config.winsor_high)
    parser.add_argument("--categorical-encoding", type=str, default=Config.categorical_encoding,
                        choices=["freq", "onehot", "drop"])
    parser.add_argument("--apply-winsor", action="store_true", default=Config.apply_winsor)
    parser.add_argument("--no-apply-winsor", action="store_false", dest="apply_winsor")
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
        drop_cols=args.drop_cols,
        winsor_low=args.winsor_low,
        winsor_high=args.winsor_high,
        categorical_encoding=args.categorical_encoding,
        apply_winsor=args.apply_winsor,
        output_json=args.output_json,
    )

    np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ref_raw = load_csv(cfg.reference_path, cfg.max_rows)
    an_raw = load_csv(cfg.analysis_path, cfg.max_rows)
    drop_cols = [c.strip() for c in cfg.drop_cols.split(",") if c.strip()]
    ref_raw, dropped = drop_columns(ref_raw, drop_cols)
    an_raw, _ = drop_columns(an_raw, drop_cols)
    num_ref = ref_raw.select_dtypes(include=[np.number])
    num_an = an_raw.select_dtypes(include=[np.number])
    cat_cols = [c for c in ref_raw.columns if c not in num_ref.columns]

    if cfg.categorical_encoding == "onehot":
        ref_enc, an_enc = encode_features(ref_raw, an_raw)
    elif cfg.categorical_encoding == "freq":
        ref_enc = num_ref.copy()
        an_enc = num_an.copy()
        for col in cat_cols:
            freq = ref_raw[col].value_counts(normalize=True)
            ref_enc[f"{col}_freq"] = ref_raw[col].map(freq).fillna(0.0)
            an_enc[f"{col}_freq"] = an_raw[col].map(freq).fillna(0.0)
    else:
        ref_enc = num_ref
        an_enc = num_an
    if ref_enc.shape[1] == 0:
        raise ValueError("No usable feature columns after encoding.")
    x_ref = ref_enc.to_numpy(dtype=np.float32)
    x_an = an_enc.to_numpy(dtype=np.float32)

    if cfg.apply_winsor:
        x_ref = winsorize_train(x_ref, x_ref, cfg.winsor_low, cfg.winsor_high)
        x_an = winsorize_train(x_ref, x_an, cfg.winsor_low, cfg.winsor_high)

    x_ref_n = normalize_train(x_ref, x_ref).astype(np.float32)
    x_an_n = normalize_train(x_ref, x_an).astype(np.float32)

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
        "features": {
            "count": int(x_ref.shape[1]),
            "dropped": dropped,
            "categorical_encoding": cfg.categorical_encoding,
            "apply_winsor": cfg.apply_winsor,
        },
    }
    report["status"] = {
        "analysis": batch_status(report["flag_rate"]["analysis"]),
        "warn_threshold": 0.10,
        "block_threshold": 0.25,
    }
    worst_idx = np.argsort(-s_an)[:5].tolist()
    report["top_worst_rows"] = {"analysis_idx": worst_idx}

    os.makedirs(os.path.dirname(cfg.output_json), exist_ok=True)
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {cfg.output_json}")


if __name__ == "__main__":
    main()
