import argparse
import csv
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
    data_path: str = "tabular/exp_tabular_fraud/data/creditcard.csv"
    seed: int = 7
    max_rows: int = 200000
    train_frac: float = 0.7
    val_frac: float = 0.15
    epochs_ae: int = 5
    epochs_clf: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    alpha_min: float = -10.0
    alpha_max: float = 1.9
    lambda_adaloss: float = 0.2
    output_json: str = "outputs/tabular_gate2_adaloss.json"


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


class TabularMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


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
            times.append(float(row[0]))
            feats.append([float(x) for x in row[0:-1]])
            labels.append(int(float(row[-1])))
    t = np.asarray(times, dtype=np.float32)
    x = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return t, x, y


def normalize_train(x_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(labels_sorted.sum(), 1e-12)
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(precision, recall))


def residual_score(model: TabularAE, x: torch.Tensor) -> torch.Tensor:
    recon = model(x)
    return (recon - x).pow(2).mean(dim=1)


def adaloss_from_residual(residual: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    a = alpha.clone()
    a = torch.where(a.abs() < eps, a + eps * torch.sign(a + 1e-9), a)
    a = torch.where((a - 2.0).abs() < eps, a - eps, a)
    b = (a - 2.0).abs().clamp_min(eps)
    return (b / a) * ((residual.pow(2) / b + 1.0).pow(a / 2.0) - 1.0)


def alpha_from_score(score: np.ndarray, alpha_min: float, alpha_max: float) -> np.ndarray:
    score = np.clip(score, -10.0, 10.0)
    s = 1.0 / (1.0 + np.exp(-score))
    a = alpha_min + (alpha_max - alpha_min) * s
    return np.clip(a, alpha_min, alpha_max).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = list(zip(torch.from_numpy(x), torch.from_numpy(y)))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular Gate2: AdaLoss on supervised CE (creditcard).")
    parser.add_argument("--data-path", type=str, default=Config.data_path)
    parser.add_argument("--max-rows", type=int, default=Config.max_rows)
    parser.add_argument("--train-frac", type=float, default=Config.train_frac)
    parser.add_argument("--val-frac", type=float, default=Config.val_frac)
    parser.add_argument("--epochs-ae", type=int, default=Config.epochs_ae)
    parser.add_argument("--epochs-clf", type=int, default=Config.epochs_clf)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--alpha-min", type=float, default=Config.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=Config.alpha_max)
    parser.add_argument("--lambda-adaloss", type=float, default=Config.lambda_adaloss)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-json", type=str, default=Config.output_json)
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data_path,
        seed=args.seed,
        max_rows=args.max_rows,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        epochs_ae=args.epochs_ae,
        epochs_clf=args.epochs_clf,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        lambda_adaloss=args.lambda_adaloss,
        output_json=args.output_json,
    )

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    t, x, y = load_creditcard(cfg.data_path, cfg.max_rows)
    order = np.argsort(t)
    x = x[order]
    y = y[order]

    n = len(x)
    n_train = int(cfg.train_frac * n)
    n_val = int(cfg.val_frac * n)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

    # Normalize using train split
    x_train_n = normalize_train(x_train, x_train)
    x_val_n = normalize_train(x_train, x_val)
    x_test_n = normalize_train(x_train, x_test)

    # Train AE on normal rows only (clean baseline)
    normal_mask = (y_train == 0)
    x_train_norm = x_train_n[normal_mask]
    ae = TabularAE(x_train_n.shape[1]).to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg.lr)
    ae.train()
    train_t = torch.from_numpy(x_train_norm).to(device)
    for epoch in range(1, cfg.epochs_ae + 1):
        perm = torch.randperm(train_t.size(0), device=device)
        total = 0.0
        for i in range(0, train_t.size(0), cfg.batch_size):
            batch = train_t[perm[i:i + cfg.batch_size]]
            recon = ae(batch)
            loss = F.mse_loss(recon, batch)
            opt_ae.zero_grad(set_to_none=True)
            loss.backward()
            opt_ae.step()
            total += loss.item()
        print(f"ae epoch {epoch:02d} | loss {total / max(1, (train_t.size(0) // cfg.batch_size)):.4f}")

    # Compute residual scores for train split (all rows)
    ae.eval()
    with torch.no_grad():
        s_train = residual_score(ae, torch.from_numpy(x_train_n).to(device)).cpu().numpy()
    score_mean = float(s_train.mean())
    score_std = float(s_train.std() + 1e-6)
    score_norm = (s_train - score_mean) / score_std
    alpha = alpha_from_score(score_norm, cfg.alpha_min, cfg.alpha_max)

    # Baseline classifier
    clf_base = TabularMLP(x_train_n.shape[1]).to(device)
    opt_base = torch.optim.Adam(clf_base.parameters(), lr=cfg.lr)
    train_loader = make_loader(x_train_n, y_train, cfg.batch_size, shuffle=True)
    for epoch in range(1, cfg.epochs_clf + 1):
        total = 0.0
        clf_base.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            logits = clf_base(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt_base.zero_grad(set_to_none=True)
            loss.backward()
            opt_base.step()
            total += loss.item()
        print(f"baseline epoch {epoch:02d} | loss {total / max(1, len(train_loader)):.4f}")

    # AdaLoss classifier
    clf_ada = TabularMLP(x_train_n.shape[1]).to(device)
    opt_ada = torch.optim.Adam(clf_ada.parameters(), lr=cfg.lr)
    alpha_t = torch.from_numpy(alpha).to(device)
    for epoch in range(1, cfg.epochs_clf + 1):
        total = 0.0
        clf_ada.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            logits = clf_ada(xb)
            ce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (ce + cfg.lambda_adaloss * adaloss_from_residual(ce, alpha_t[:ce.size(0)])).mean()
            opt_ada.zero_grad(set_to_none=True)
            loss.backward()
            opt_ada.step()
            total += loss.item()
        print(f"adaloss epoch {epoch:02d} | loss {total / max(1, len(train_loader)):.4f}")

    def eval_pr(model: TabularMLP, x_eval: np.ndarray, y_eval: np.ndarray) -> float:
        model.eval()
        xs = torch.from_numpy(x_eval).to(device)
        with torch.no_grad():
            logits = model(xs).cpu().numpy()
        logits = np.clip(logits, -20.0, 20.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return pr_auc(probs.reshape(-1), y_eval.astype(np.int64))

    pr_val_base = eval_pr(clf_base, x_val_n, y_val)
    pr_test_base = eval_pr(clf_base, x_test_n, y_test)
    pr_val_ada = eval_pr(clf_ada, x_val_n, y_val)
    pr_test_ada = eval_pr(clf_ada, x_test_n, y_test)

    report = {
        "config": asdict(cfg),
        "device": str(device),
        "sizes": {
            "train": int(len(x_train_n)),
            "val": int(len(x_val_n)),
            "test": int(len(x_test_n)),
        },
        "alpha": {
            "mean": float(alpha.mean()),
            "p10": float(np.quantile(alpha, 0.1)),
            "p50": float(np.quantile(alpha, 0.5)),
            "p90": float(np.quantile(alpha, 0.9)),
        },
        "pr_auc": {
            "baseline": {"val": float(pr_val_base), "test": float(pr_test_base)},
            "adaloss": {"val": float(pr_val_ada), "test": float(pr_test_ada)},
        },
    }

    os.makedirs(os.path.dirname(cfg.output_json), exist_ok=True)
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {cfg.output_json}")


if __name__ == "__main__":
    main()
