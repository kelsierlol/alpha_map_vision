import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def make_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cmapss(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    unit_ids = data[:, 0].astype(int)
    sensors = data[:, 5:]
    return unit_ids, sensors


def build_sequences(unit_ids: np.ndarray, sensors: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    sequences = []
    for uid in np.unique(unit_ids):
        series = sensors[unit_ids == uid]
        if series.shape[0] < seq_len:
            continue
        for start in range(0, series.shape[0] - seq_len + 1, stride):
            sequences.append(series[start:start + seq_len])
    if not sequences:
        raise ValueError("No sequences built. Check seq_len/stride.")
    return np.stack(sequences, axis=0)


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class ConvAE1D(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class AlphaHead1D(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simulate_dropouts(
    x: torch.Tensor,
    rng: torch.Generator,
    drop_prob: float,
    drop_len: int,
    drop_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, channels, length = x.shape
    corrupt = x.clone()
    mask = torch.zeros((bsz, 1, length), device=x.device)
    for i in range(bsz):
        if torch.rand(1, generator=rng).item() > drop_prob:
            continue
        start = torch.randint(0, max(1, length - drop_len), (1,), generator=rng).item()
        end = min(length, start + drop_len)
        corrupt[i, :, start:end] = drop_value
        mask[i, 0, start:end] = 1.0
    return corrupt, mask


def simulate_stuck_at(
    x: torch.Tensor,
    rng: torch.Generator,
    stuck_prob: float,
    stuck_len: int,
    stuck_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, channels, length = x.shape
    corrupt = x.clone()
    mask = torch.zeros((bsz, 1, length), device=x.device)
    for i in range(bsz):
        if torch.rand(1, generator=rng).item() > stuck_prob:
            continue
        start = torch.randint(0, max(1, length - stuck_len), (1,), generator=rng).item()
        end = min(length, start + stuck_len)
        corrupt[i, :, start:end] = stuck_value
        mask[i, 0, start:end] = 1.0
    return corrupt, mask


def simulate_drift(
    x: torch.Tensor,
    rng: torch.Generator,
    drift_prob: float,
    drift_len: int,
    drift_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, channels, length = x.shape
    corrupt = x.clone()
    mask = torch.zeros((bsz, 1, length), device=x.device)
    for i in range(bsz):
        if torch.rand(1, generator=rng).item() > drift_prob:
            continue
        start = torch.randint(0, max(1, length - drift_len), (1,), generator=rng).item()
        end = min(length, start + drift_len)
        drift = torch.linspace(0, drift_scale, end - start, device=x.device).view(1, -1)
        corrupt[i, :, start:end] += drift
        mask[i, 0, start:end] = 1.0
    return corrupt, mask


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
    parser = argparse.ArgumentParser(description="Gate 1 metrics on CMAPSS (dropout masks).")
    parser.add_argument("--data-path", type=str, default="../data/train_FD001.txt")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-ae", type=int, default=5)
    parser.add_argument("--epochs-alpha", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--drop-prob", type=float, default=0.7)
    parser.add_argument("--drop-len", type=int, default=8)
    parser.add_argument("--drop-value", type=float, default=-3.0)
    parser.add_argument("--train-mode", type=str, default="dropout", choices=["dropout", "stuck", "drift", "mixed"])
    parser.add_argument("--eval-mode", type=str, default="dropout", choices=["dropout", "stuck", "drift"])
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="outputs/gate1_cmapss_metrics.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = make_device()

    unit_ids, sensors = load_cmapss(args.data_path)
    seqs = build_sequences(unit_ids, sensors, args.seq_len, args.stride)
    if args.max_samples:
        seqs = seqs[: args.max_samples]

    mean = seqs.mean(axis=(0, 1), keepdims=True)
    std = seqs.std(axis=(0, 1), keepdims=True) + 1e-6
    seqs = (seqs - mean) / std

    dataset = SequenceDataset(seqs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    channels = seqs.shape[2]
    ae = ConvAE1D(channels=channels, hidden=64).to(device)
    alpha_head = AlphaHead1D(in_ch=1).to(device)

    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    ae.train()
    for epoch in range(1, args.epochs_ae + 1):
        total = 0.0
        for x in loader:
            x = x.to(device).permute(0, 2, 1)
            recon = ae(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"ae epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")

    for p in ae.parameters():
        p.requires_grad = False
    ae.eval()
    opt_a = torch.optim.Adam(alpha_head.parameters(), lr=args.lr)
    alpha_head.train()
    rng = torch.Generator().manual_seed(args.seed + 1)

    def make_corrupt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if args.train_mode == "dropout":
            return simulate_dropouts(x, rng, args.drop_prob, args.drop_len, args.drop_value)
        if args.train_mode == "stuck":
            return simulate_stuck_at(x, rng, args.drop_prob, args.drop_len, args.drop_value)
        if args.train_mode == "drift":
            return simulate_drift(x, rng, args.drop_prob, args.drop_len, args.drift_scale)
        # mixed: choose a corruption type per batch
        choice = torch.randint(0, 3, (1,), generator=rng).item()
        if choice == 0:
            return simulate_dropouts(x, rng, args.drop_prob, args.drop_len, args.drop_value)
        if choice == 1:
            return simulate_stuck_at(x, rng, args.drop_prob, args.drop_len, args.drop_value)
        return simulate_drift(x, rng, args.drop_prob, args.drop_len, args.drift_scale)

    for epoch in range(1, args.epochs_alpha + 1):
        total = 0.0
        for x in loader:
            x = x.to(device).permute(0, 2, 1)
            corrupt, mask = make_corrupt(x)
            with torch.no_grad():
                recon = ae(corrupt)
            resid = (recon - corrupt).pow(2).mean(dim=1, keepdim=True).detach()
            logits = alpha_head(resid)
            pos = mask.sum()
            neg = mask.numel() - pos
            pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_a.step()
            total += loss.item()
        print(f"alpha epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")

    # Metrics
    ae.eval()
    alpha_head.eval()
    scores = []
    labels = []
    def make_eval_corrupt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if args.eval_mode == "dropout":
            return simulate_dropouts(x, rng, args.drop_prob, args.drop_len, args.drop_value)
        if args.eval_mode == "stuck":
            return simulate_stuck_at(x, rng, args.drop_prob, args.drop_len, args.drop_value)
        return simulate_drift(x, rng, args.drop_prob, args.drop_len, args.drift_scale)

    with torch.no_grad():
        for x in loader:
            x = x.to(device).permute(0, 2, 1)
            corrupt, mask = make_eval_corrupt(x)
            recon = ae(corrupt)
            resid = (recon - corrupt).pow(2).mean(dim=1, keepdim=True)
            logits = alpha_head(resid)
            score_map = torch.sigmoid(logits).cpu().numpy()
            scores.append(score_map.reshape(score_map.shape[0], -1))
            labels.append(mask.cpu().numpy().reshape(mask.shape[0], -1))

    scores_np = np.concatenate(scores, axis=0).reshape(-1)
    labels_np = np.concatenate(labels, axis=0).reshape(-1)
    report = {
        "config": vars(args),
        "metrics": {
            "pr_auc": float(pr_auc(scores_np, labels_np)),
            "iou_k": float(iou_at_k(scores_np, labels_np)),
            "fpr_90": float(fpr_at_recall(scores_np, labels_np, 0.9)),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
