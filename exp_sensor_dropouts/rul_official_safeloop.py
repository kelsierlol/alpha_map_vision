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


def load_rul(path: str) -> np.ndarray:
    return np.loadtxt(path).astype(np.float32)


def build_sequences(
    unit_ids: np.ndarray,
    cycles: np.ndarray,
    sensors: np.ndarray,
    seq_len: int,
    stride: int,
    rul_targets: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for uid in np.unique(unit_ids):
        series = sensors[unit_ids == uid]
        cycles_u = cycles[unit_ids == uid]
        max_cycle = cycles_u.max()
        if series.shape[0] < seq_len:
            continue
        for start in range(0, series.shape[0] - seq_len + 1, stride):
            end = start + seq_len
            sequences.append(series[start:end])
            if rul_targets is None:
                rul = max_cycle - cycles_u[end - 1]
            else:
                rul = rul_targets[uid - 1]
            targets.append(rul)
    return np.stack(sequences), np.array(targets, dtype=np.float32)


class RULDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TCNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.net(x)
        h = h.mean(dim=2)
        return self.fc(h).squeeze(1)


class AlphaHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(1)


def eval_rmse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
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


def corrupt_batch(xb: torch.Tensor, drop_prob: float, drop_len: int, drop_value: float) -> torch.Tensor:
    xb = xb.clone()
    bsz, steps, dims = xb.shape
    for i in range(bsz):
        if torch.rand(1).item() > drop_prob:
            continue
        start = torch.randint(0, max(1, steps - drop_len), (1,)).item()
        end = min(steps, start + drop_len)
        xb[i, start:end, :] = drop_value
    return xb


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS official split with SafeLoop weighting.")
    parser.add_argument("--train-path", type=str, default="../data/train_FD001.txt")
    parser.add_argument("--test-path", type=str, default="../data/test_FD001.txt")
    parser.add_argument("--rul-path", type=str, default="../data/RUL_FD001.txt")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--drop-prob", type=float, default=0.7)
    parser.add_argument("--drop-len", type=int, default=8)
    parser.add_argument("--drop-value", type=float, default=-3.0)
    parser.add_argument("--alpha-strength", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = make_device()

    unit_ids, cycles, sensors = load_cmapss(args.train_path)
    x_train, y_train = build_sequences(unit_ids, cycles, sensors, args.seq_len, args.stride)

    unit_ids_t, cycles_t, sensors_t = load_cmapss(args.test_path)
    rul_targets = load_rul(args.rul_path)
    x_test, y_test = build_sequences(
        unit_ids_t, cycles_t, sensors_t, args.seq_len, args.stride, rul_targets=rul_targets
    )

    if args.max_train:
        x_train = x_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_test:
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_loader = DataLoader(RULDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(RULDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = TCNRegressor(in_dim=x_train.shape[2], hidden=128).to(device)
    alpha_head = AlphaHead(in_dim=x_train.shape[2], hidden=64).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(alpha_head.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        alpha_head.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            feat = xb.mean(dim=1)
            alpha_logits = alpha_head(feat)
            weights = torch.sigmoid(-args.alpha_strength * alpha_logits)
            loss = ((preds - yb) ** 2 * weights).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        clean_rmse = eval_rmse(model, test_loader, device)
        # Corrupted test RMSE
        model.eval()
        total_c = 0.0
        count_c = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = corrupt_batch(xb, args.drop_prob, args.drop_len, args.drop_value).to(device)
                yb = yb.to(device)
                preds = model(xb)
                total_c += ((preds - yb) ** 2).sum().item()
                count_c += yb.numel()
        corrupt_rmse = float(np.sqrt(total_c / max(count_c, 1)))

        print(
            f"epoch {epoch:02d} | weighted_mse {total / max(1, len(train_loader)):.4f} | "
            f"clean_rmse {clean_rmse:.2f} | corrupt_rmse {corrupt_rmse:.2f}"
        )

    print("\nSummary (final epoch)")
    print("clean_rmse | corrupt_rmse")
    print(f"{clean_rmse:.2f} | {corrupt_rmse:.2f}")


if __name__ == "__main__":
    main()
