import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from scripts.alpha_map_cifar import TinyUNet2D, corrupt_batch, set_seed
from scripts.alpha_map_cifar_twophase import AlphaHead2D


def make_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_unet(model: TinyUNet2D, loader: DataLoader, device: torch.device, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"unet epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")


def train_alpha(
    unet: TinyUNet2D,
    alpha_head: AlphaHead2D,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    for p in unet.parameters():
        p.requires_grad = False
    unet.eval()

    opt = torch.optim.Adam(alpha_head.parameters(), lr=lr)
    alpha_head.train()
    rng = torch.Generator().manual_seed(123)
    modes = ("occlusion", "blur")

    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            corrupt, mask = corrupt_batch(x, rng, modes=modes)
            with torch.no_grad():
                recon = unet(corrupt)
            resid = (recon - corrupt).pow(2).mean(dim=1, keepdim=True).detach()

            logits = alpha_head(resid)
            pos = mask.sum()
            neg = mask.numel() - pos
            pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"alpha epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeLoop OOD demo (Cleanlab-adjacent).")
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs-unet", type=int, default=3)
    parser.add_argument("--epochs-alpha", type=int, default=3)
    parser.add_argument("--lr-unet", type=float, default=1e-3)
    parser.add_argument("--lr-alpha", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    device = make_device()

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tf)

    animal_labels = {2, 3, 4, 5, 6, 7}  # bird, cat, deer, dog, frog, horse
    train_idx: List[int] = [i for i, (_, y) in enumerate(train_set) if y in animal_labels]
    train_idx = train_idx[: args.max_samples]

    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    unet = TinyUNet2D().to(device)
    alpha_head = AlphaHead2D(in_ch=1).to(device)

    train_unet(unet, train_loader, device, args.epochs_unet, args.lr_unet)
    train_alpha(unet, alpha_head, train_loader, device, args.epochs_alpha, args.lr_alpha)

    # OOD scoring: non-animal classes are OOD
    ood_labels = {0, 1, 8, 9}  # airplane, automobile, ship, truck
    scores = []
    labels = []
    unet.eval()
    alpha_head.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon = unet(x)
            resid = (recon - x).pow(2).mean(dim=1, keepdim=True)
            logits = alpha_head(resid)
            score = torch.sigmoid(logits).mean(dim=(1, 2, 3)).cpu().numpy()
            scores.append(score)
            labels.append(np.isin(y.numpy(), list(ood_labels)).astype(np.int32))

    scores_np = np.concatenate(scores)
    labels_np = np.concatenate(labels)
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        print("sklearn not available; cannot compute AUROC.")
        return

    auroc = roc_auc_score(labels_np, scores_np)
    print(f"OOD AUROC (animals vs non-animals): {auroc:.4f}")


if __name__ == "__main__":
    main()
