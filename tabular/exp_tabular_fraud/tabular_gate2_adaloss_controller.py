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
    train_quantile: float = 0.6
    holdout_quantile: float = 0.7
    mid_quantile: float = 0.8
    epochs_ae: int = 5
    epochs_clf: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    alpha_min: float = -5.0
    alpha_max: float = 1.9
    alpha0: float = 1.0
    lambda_adaloss: float = 0.2
    reg_lambda: float = 0.01
    output_json: str = "outputs/tabular_gate2_adaloss_controller.json"


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


class AlphaController(nn.Module):
    def __init__(self, hidden: int, alpha_min: float, alpha_max: float, alpha0: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha0 = alpha0

    def forward(self, resid_mag: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([resid_mag, t_norm], dim=1)
        a = self.net(x) + self.alpha0
        return torch.clamp(a, self.alpha_min, self.alpha_max)


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


def barron_loss(residual: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    while alpha.dim() < residual.dim():
        alpha = alpha.unsqueeze(-1)
    a = alpha
    a_safe = torch.where(a.abs() < eps, torch.full_like(a, eps), a)
    denom = (a - 2.0).abs()
    denom_safe = torch.where(denom < eps, torch.full_like(denom, eps), denom)
    r2 = residual.pow(2)
    inner = r2 / denom_safe + 1.0
    pow_term = inner.pow(a_safe / 2.0) - 1.0
    return (denom_safe / a_safe) * pow_term


def make_loader(x: np.ndarray, y: np.ndarray, t_norm: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = list(zip(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(t_norm)))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular Gate2 AdaLoss with learned alpha controller.")
    parser.add_argument("--data-path", type=str, default=Config.data_path)
    parser.add_argument("--max-rows", type=int, default=Config.max_rows)
    parser.add_argument("--train-quantile", type=float, default=Config.train_quantile)
    parser.add_argument("--holdout-quantile", type=float, default=Config.holdout_quantile)
    parser.add_argument("--mid-quantile", type=float, default=Config.mid_quantile)
    parser.add_argument("--epochs-ae", type=int, default=Config.epochs_ae)
    parser.add_argument("--epochs-clf", type=int, default=Config.epochs_clf)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--alpha-min", type=float, default=Config.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=Config.alpha_max)
    parser.add_argument("--alpha0", type=float, default=Config.alpha0)
    parser.add_argument("--lambda-adaloss", type=float, default=Config.lambda_adaloss)
    parser.add_argument("--reg-lambda", type=float, default=Config.reg_lambda)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-json", type=str, default=Config.output_json)
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data_path,
        seed=args.seed,
        max_rows=args.max_rows,
        train_quantile=args.train_quantile,
        holdout_quantile=args.holdout_quantile,
        mid_quantile=args.mid_quantile,
        epochs_ae=args.epochs_ae,
        epochs_clf=args.epochs_clf,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha0=args.alpha0,
        lambda_adaloss=args.lambda_adaloss,
        reg_lambda=args.reg_lambda,
        output_json=args.output_json,
    )

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    t, x, y = load_creditcard(cfg.data_path, cfg.max_rows)
    # Use normal transactions for slicing
    normal = (y == 0)
    t_n = t[normal]
    x_n = x[normal]
    y_n = y[normal]

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

    # Normalize using train split
    x_train_n = normalize_train(x_train, x_train)
    x_holdout_n = normalize_train(x_train, x_holdout)
    x_mid_n = normalize_train(x_train, x_mid)
    x_late_n = normalize_train(x_train, x_late)

    # AE on clean train slice
    ae = TabularAE(x_train_n.shape[1]).to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg.lr)
    ae.train()
    train_t = torch.from_numpy(x_train_n).to(device)
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

    # Train classifier on full labeled data (time-ordered)
    # Rebuild splits on full data to avoid leakage across time
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]

    n = len(x)
    n_train = int(cfg.train_quantile * n)
    n_holdout = int(cfg.holdout_quantile * n)
    n_mid = int(cfg.mid_quantile * n)
    x_train_all, y_train_all, t_train_all = x[:n_train], y[:n_train], t[:n_train]
    x_holdout_all, y_holdout_all, t_holdout_all = x[n_train:n_holdout], y[n_train:n_holdout], t[n_train:n_holdout]
    x_mid_all, y_mid_all, t_mid_all = x[n_holdout:n_mid], y[n_holdout:n_mid], t[n_holdout:n_mid]
    x_late_all, y_late_all, t_late_all = x[n_mid:], y[n_mid:], t[n_mid:]

    # Normalize with train stats
    x_train_all_n = normalize_train(x_train_all, x_train_all)
    x_holdout_all_n = normalize_train(x_train_all, x_holdout_all)
    x_mid_all_n = normalize_train(x_train_all, x_mid_all)
    x_late_all_n = normalize_train(x_train_all, x_late_all)

    t_min, t_max = t_train_all.min(), t_train_all.max()
    def norm_t(t_arr: np.ndarray) -> np.ndarray:
        denom = max(float(t_max - t_min), 1e-6)
        return ((t_arr - t_min) / denom).astype(np.float32)

    train_loader = make_loader(x_train_all_n, y_train_all, norm_t(t_train_all), cfg.batch_size, shuffle=True)
    holdout_loader = make_loader(x_holdout_all_n, y_holdout_all, norm_t(t_holdout_all), cfg.batch_size, shuffle=False)
    mid_loader = make_loader(x_mid_all_n, y_mid_all, norm_t(t_mid_all), cfg.batch_size, shuffle=False)
    late_loader = make_loader(x_late_all_n, y_late_all, norm_t(t_late_all), cfg.batch_size, shuffle=False)

    # Baseline classifier
    clf_base = TabularMLP(x_train_all_n.shape[1]).to(device)
    opt_base = torch.optim.Adam(clf_base.parameters(), lr=cfg.lr)
    for epoch in range(1, cfg.epochs_clf + 1):
        total = 0.0
        clf_base.train()
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            logits = clf_base(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt_base.zero_grad(set_to_none=True)
            loss.backward()
            opt_base.step()
            total += loss.item()
        print(f"baseline epoch {epoch:02d} | loss {total / max(1, len(train_loader)):.4f}")

    # AdaLoss classifier + controller
    clf_ada = TabularMLP(x_train_all_n.shape[1]).to(device)
    controller = AlphaController(hidden=16, alpha_min=cfg.alpha_min, alpha_max=cfg.alpha_max, alpha0=cfg.alpha0).to(device)
    opt_ada = torch.optim.Adam(list(clf_ada.parameters()) + list(controller.parameters()), lr=cfg.lr)
    ae.eval()
    for epoch in range(1, cfg.epochs_clf + 1):
        total = 0.0
        clf_ada.train()
        controller.train()
        for xb, yb, tb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            tb = tb.to(device).unsqueeze(1)
            with torch.no_grad():
                recon = ae(xb)
                resid_mag = (recon - xb).pow(2).mean(dim=1, keepdim=True).detach()
            alpha = controller(resid_mag, tb)
            logits = clf_ada(xb)
            ce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            ada = barron_loss(ce, alpha).squeeze()
            reg = (alpha - cfg.alpha0).abs().mean()
            loss = (ce + cfg.lambda_adaloss * ada).mean() + cfg.reg_lambda * reg
            opt_ada.zero_grad(set_to_none=True)
            loss.backward()
            opt_ada.step()
            total += loss.item()
        print(f"adaloss epoch {epoch:02d} | loss {total / max(1, len(train_loader)):.4f}")

    def eval_pr(model: TabularMLP, loader: torch.utils.data.DataLoader) -> float:
        model.eval()
        probs = []
        labels = []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                logits = np.clip(logits, -20.0, 20.0)
                p = 1.0 / (1.0 + np.exp(-logits))
                probs.append(p)
                labels.append(yb.numpy())
        probs = np.concatenate(probs).reshape(-1)
        labels = np.concatenate(labels).reshape(-1)
        return pr_auc(probs, labels.astype(np.int64))

    pr_holdout_base = eval_pr(clf_base, holdout_loader)
    pr_mid_base = eval_pr(clf_base, mid_loader)
    pr_late_base = eval_pr(clf_base, late_loader)
    pr_holdout_ada = eval_pr(clf_ada, holdout_loader)
    pr_mid_ada = eval_pr(clf_ada, mid_loader)
    pr_late_ada = eval_pr(clf_ada, late_loader)

    report = {
        "config": asdict(cfg),
        "device": str(device),
        "sizes": {
            "train": int(len(x_train_all_n)),
            "holdout": int(len(x_holdout_all_n)),
            "mid": int(len(x_mid_all_n)),
            "late": int(len(x_late_all_n)),
        },
        "pr_auc": {
            "baseline": {"holdout": float(pr_holdout_base), "mid": float(pr_mid_base), "late": float(pr_late_base)},
            "adaloss": {"holdout": float(pr_holdout_ada), "mid": float(pr_mid_ada), "late": float(pr_late_ada)},
        },
    }

    os.makedirs(os.path.dirname(cfg.output_json), exist_ok=True)
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {cfg.output_json}")


if __name__ == "__main__":
    main()
