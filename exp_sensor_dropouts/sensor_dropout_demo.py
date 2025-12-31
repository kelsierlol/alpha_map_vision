import argparse
import json
import os
import sys
from typing import List, Tuple

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
    sensors = data[:, 5:]  # 21 sensors
    return unit_ids, sensors


def build_sequences(
    unit_ids: np.ndarray,
    sensors: np.ndarray,
    seq_len: int,
    stride: int,
) -> np.ndarray:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, channels, length = x.shape
    corrupt = x.clone()
    mask = torch.zeros((bsz, 1, length), device=x.device)

    for i in range(bsz):
        if torch.rand(1, generator=rng).item() > drop_prob:
            continue
        start = torch.randint(0, max(1, length - drop_len), (1,), generator=rng).item()
        end = min(length, start + drop_len)
        corrupt[i, :, start:end] = 0.0
        mask[i, 0, start:end] = 1.0
    return corrupt, mask


def summarize_batch(alpha: torch.Tensor, mask: torch.Tensor) -> dict:
    coverage = float(mask.mean().item())
    alpha_mean = float(alpha.mean().item())
    return {"coverage": coverage, "alpha_mean": alpha_mean}


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
    parser = argparse.ArgumentParser(description="SafeLoop sensor dropout demo (CMAPSS).")
    parser.add_argument("--data-path", type=str, default="../data/train_FD001.txt")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-ae", type=int, default=10)
    parser.add_argument("--epochs-alpha", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--drop-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="outputs/sensor_dropout_report.json")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=30)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = make_device()

    unit_ids, sensors = load_cmapss(args.data_path)
    seqs = build_sequences(unit_ids, sensors, args.seq_len, args.stride)
    if args.max_samples:
        seqs = seqs[: args.max_samples]

    # Normalize
    mean = seqs.mean(axis=(0, 1), keepdims=True)
    std = seqs.std(axis=(0, 1), keepdims=True) + 1e-6
    seqs = (seqs - mean) / std

    dataset = SequenceDataset(seqs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    channels = seqs.shape[2]
    ae = ConvAE1D(channels=channels, hidden=64).to(device)
    alpha_head = AlphaHead1D(in_ch=1).to(device)

    # Train AE on clean sequences
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    ae.train()
    for epoch in range(1, args.epochs_ae + 1):
        total = 0.0
        for x in loader:
            x = x.to(device).permute(0, 2, 1)  # (B,C,L)
            recon = ae(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"ae epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")

    # Train alpha head on simulated dropouts
    for p in ae.parameters():
        p.requires_grad = False
    ae.eval()
    opt_a = torch.optim.Adam(alpha_head.parameters(), lr=args.lr)
    alpha_head.train()
    rng = torch.Generator().manual_seed(args.seed + 1)

    for epoch in range(1, args.epochs_alpha + 1):
        total = 0.0
        for x in loader:
            x = x.to(device).permute(0, 2, 1)
            corrupt, mask = simulate_dropouts(x, rng, args.drop_prob, args.drop_len)
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

    # Generate a small report
    ae.eval()
    alpha_head.eval()
    batch_metrics = []
    scores = []
    labels = []
    seq_scores = []
    with torch.no_grad():
        for i, x in enumerate(loader):
            x = x.to(device).permute(0, 2, 1)
            corrupt, mask = simulate_dropouts(x, rng, args.drop_prob, args.drop_len)
            recon = ae(corrupt)
            resid = (recon - corrupt).pow(2).mean(dim=1, keepdim=True)
            alpha = torch.sigmoid(alpha_head(resid))
            batch_metrics.append(summarize_batch(alpha, mask))
            score_map = (1.0 - alpha).cpu().numpy()
            scores.append(score_map.reshape(score_map.shape[0], -1))
            labels.append(mask.cpu().numpy().reshape(mask.shape[0], -1))
            seq_scores.extend(score_map.mean(axis=(1, 2)))
            if i + 1 >= args.eval_batches:
                break

    scores_np = np.concatenate(scores, axis=0).reshape(-1)
    labels_np = np.concatenate(labels, axis=0).reshape(-1)
    pr = pr_auc(scores_np, labels_np)
    iou = iou_at_k(scores_np, labels_np)
    fpr = fpr_at_recall(scores_np, labels_np, 0.9)
    top_idx = [int(x) for x in np.argsort(seq_scores)[-args.top_n:][::-1]]

    report = {
        "config": vars(args),
        "device": str(device),
        "samples": int(seqs.shape[0]),
        "channels": int(channels),
        "avg_coverage": float(np.mean([b["coverage"] for b in batch_metrics])),
        "avg_alpha_mean": float(np.mean([b["alpha_mean"] for b in batch_metrics])),
        "metrics": {
            "pr_auc": float(pr),
            "iou_k": float(iou),
            "fpr_90": float(fpr),
        },
        "top_n_sequences": top_idx,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
