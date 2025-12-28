import argparse
import os
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from alpha_map_cifar import (
    AlphaController2D,
    TinyUNet2D,
    corrupt_batch,
    local_redundancy,
    pr_auc,
    set_seed,
)


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


def collect_scores(
    loader: Iterable,
    model: TinyUNet2D,
    controller: AlphaController2D,
    rng: torch.Generator,
    device: torch.device,
    corruption_modes: Tuple[str, ...],
    redundancy_window: int,
    max_batches: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        corrupt, mask = corrupt_batch(x, rng, modes=corruption_modes)
        with torch.no_grad():
            recon = model(corrupt)
            resid = (recon - corrupt).abs().mean(dim=1, keepdim=True)
            redundancy = local_redundancy(corrupt, redundancy_window).mean(dim=1, keepdim=True)
            alpha = controller(resid, redundancy)
        score = (-(alpha - controller.alpha0)).cpu().numpy().reshape(-1)
        scores.append(score)
        labels.append(mask.cpu().numpy().reshape(-1))
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha map evaluation + generalization test (CIFAR-10).")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-unet", type=int, default=3)
    parser.add_argument("--epochs-alpha", type=int, default=3)
    parser.add_argument("--lr-unet", type=float, default=1e-3)
    parser.add_argument("--lr-alpha", type=float, default=5e-4)
    parser.add_argument("--redundancy-window", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--weights-dir", type=str, default="weights")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    idx = torch.randperm(len(train_set))[: args.max_samples]
    subset = Subset(train_set, idx.tolist())
    train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TinyUNet2D().to(device)
    controller = AlphaController2D().to(device)

    # Stage 1: Train UNet reconstruction
    opt_unet = torch.optim.Adam(model.parameters(), lr=args.lr_unet)
    model.train()
    for epoch in range(1, args.epochs_unet + 1):
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = F.l1_loss(recon, x)
            opt_unet.zero_grad(set_to_none=True)
            loss.backward()
            opt_unet.step()
            total += loss.item()
        print(f"unet epoch {epoch:02d} | loss {total / max(1, len(train_loader)):.4f}")

    # Stage 2: Train alpha on occlusion + blur
    opt_alpha = torch.optim.Adam(controller.parameters(), lr=args.lr_alpha)
    controller.train()
    model.eval()
    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    train_modes = ("occlusion", "blur")

    for epoch in range(1, args.epochs_alpha + 1):
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            corrupt, mask = corrupt_batch(x, rng, modes=train_modes)
            with torch.no_grad():
                recon = model(corrupt)
            resid = (recon - corrupt).abs().mean(dim=1, keepdim=True).detach()
            redundancy = local_redundancy(corrupt, args.redundancy_window).mean(dim=1, keepdim=True).detach()
            alpha = controller(resid, redundancy)
            score = torch.sigmoid(-(alpha - controller.alpha0) / 0.2)
            loss = F.binary_cross_entropy(score, mask)
            opt_alpha.zero_grad(set_to_none=True)
            loss.backward()
            opt_alpha.step()
            total += loss.item()
        print(f"alpha epoch {epoch:02d} | loss {total / max(1, len(train_loader)):.4f}")

    if args.save_weights:
        os.makedirs(args.weights_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.weights_dir, "unet.pth"))
        torch.save(controller.state_dict(), os.path.join(args.weights_dir, "alpha_controller.pth"))
        print(f"Saved weights to {args.weights_dir}")

    controller.eval()
    model.eval()

    # Evaluation on seen corruption types
    scores, labels = collect_scores(
        test_loader,
        model,
        controller,
        rng,
        device,
        train_modes,
        args.redundancy_window,
        args.eval_batches,
    )
    print("Seen corruptions (occlusion+blur)")
    print(f"PR-AUC: {pr_auc(scores, labels):.4f}")
    print(f"IoU@k%: {iou_at_k(scores, labels):.4f}")
    print(f"FPR@90% recall: {fpr_at_recall(scores, labels, 0.9):.4f}")

    # Generalization on unseen corruption types
    test_modes = ("saltpepper", "copy")
    scores_u, labels_u = collect_scores(
        test_loader,
        model,
        controller,
        rng,
        device,
        test_modes,
        args.redundancy_window,
        args.eval_batches,
    )
    print("Unseen corruptions (saltpepper+copy)")
    print(f"PR-AUC: {pr_auc(scores_u, labels_u):.4f}")
    print(f"IoU@k%: {iou_at_k(scores_u, labels_u):.4f}")
    print(f"FPR@90% recall: {fpr_at_recall(scores_u, labels_u, 0.9):.4f}")


if __name__ == "__main__":
    main()
