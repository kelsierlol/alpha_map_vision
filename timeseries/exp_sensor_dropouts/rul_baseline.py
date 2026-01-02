import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def make_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cmapss(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    unit_ids = data[:, 0].astype(int)
    cycles = data[:, 1].astype(int)
    sensors = data[:, 5:]
    return unit_ids, cycles, sensors


def build_rul_sequences(
    unit_ids: np.ndarray,
    cycles: np.ndarray,
    sensors: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    rul_targets = []
    for uid in np.unique(unit_ids):
        series = sensors[unit_ids == uid]
        cycles_u = cycles[unit_ids == uid]
        max_cycle = cycles_u.max()
        if series.shape[0] < seq_len:
            continue
        for start in range(0, series.shape[0] - seq_len + 1, stride):
            end = start + seq_len
            sequences.append(series[start:end])
            rul = max_cycle - cycles_u[end - 1]
            rul_targets.append(rul)
    return np.stack(sequences), np.array(rul_targets, dtype=np.float32)


class RULDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS RUL baseline (LSTM).")
    parser.add_argument("--data-path", type=str, default="../data/train_FD001.txt")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--drop-len", type=int, default=5)
    parser.add_argument("--drop-value", type=float, default=-3.0)
    parser.add_argument("--val-drop-prob", type=float, default=1.0)
    parser.add_argument("--val-drop-len", type=int, default=10)
    parser.add_argument("--val-drop-channels", type=int, default=3)
    parser.add_argument("--val-drop-full", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = make_device()

    unit_ids, cycles, sensors = load_cmapss(args.data_path)
    x, y = build_rul_sequences(unit_ids, cycles, sensors, args.seq_len, args.stride)
    if args.max_samples:
        x = x[: args.max_samples]
        y = y[: args.max_samples]

    # Normalize per sensor
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True) + 1e-6
    x = (x - mean) / std

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(x))
    val_size = int(len(x) * args.val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    def corrupt_np(xb: np.ndarray) -> np.ndarray:
        xb = xb.copy()
        for i in range(xb.shape[0]):
            if rng.random() > args.val_drop_prob:
                continue
            if args.val_drop_full:
                xb[i, :, :] = args.drop_value
            else:
                start = rng.integers(0, max(1, xb.shape[1] - args.val_drop_len))
                end = min(xb.shape[1], start + args.val_drop_len)
                xb[i, start:end, :] = args.drop_value
                if args.val_drop_channels > 0:
                    ch = rng.choice(xb.shape[2], size=min(args.val_drop_channels, xb.shape[2]), replace=False)
                    xb[i, :, ch] = args.drop_value
        return xb

    x_val_corrupt = corrupt_np(x_val)

    train_loader = DataLoader(
        RULDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        RULDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    val_corrupt_loader = DataLoader(
        RULDataset(x_val_corrupt, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = LSTMRegressor(in_dim=x.shape[2], hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    def eval_rmse(loader: DataLoader) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                total += ((preds - yb) ** 2).sum().item()
                count += yb.numel()
        return float(np.sqrt(total / max(count, 1)))

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        val_rmse = eval_rmse(val_loader)
        # Corrupted val RMSE (fixed corruption)
        model.eval()
        total_c = 0.0
        count_c = 0
        with torch.no_grad():
            for xb, yb in val_corrupt_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                total_c += ((preds - yb) ** 2).sum().item()
                count_c += yb.numel()
        val_corrupt_rmse = float(np.sqrt(total_c / max(count_c, 1)))
        print(
            f"epoch {epoch:02d} | mse {total / max(1, len(train_loader)):.4f} | "
            f"val_rmse {val_rmse:.2f} | val_corrupt_rmse {val_corrupt_rmse:.2f}"
        )


if __name__ == "__main__":
    main()
