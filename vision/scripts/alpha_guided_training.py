import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from vision.scripts.alpha_map_cifar import TinyUNet2D, corrupt_batch, set_seed
from vision.scripts.alpha_map_cifar_twophase import AlphaHead2D


def make_resnet18(num_classes: int = 10) -> torch.nn.Module:
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def topk_mask(score: torch.Tensor, k: float) -> torch.Tensor:
    bsz = score.size(0)
    flat = score.view(bsz, -1)
    total = flat.size(1)
    k_count = max(1, int(round(k * total)))
    _, idx = torch.topk(flat, k_count, dim=1, largest=True, sorted=False)
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(1, idx, 1.0)
    return mask_flat.view_as(score)


def dilate(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)


def repair_batch(
    x: torch.Tensor,
    unet: TinyUNet2D,
    alpha_head: AlphaHead2D,
    rng: torch.Generator,
    modes: Tuple[str, ...],
    topk: float,
    dilate_k: int,
    oracle: bool = False,
    blend: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    corrupt, mask_gt = corrupt_batch(x, rng, modes=modes)
    with torch.no_grad():
        recon = unet(corrupt)
        resid = (recon - corrupt).pow(2).mean(dim=1, keepdim=True)
        if oracle:
            mask = mask_gt
        else:
            logits = alpha_head(resid)
            score = torch.sigmoid(logits)
            mask = topk_mask(score, topk)
        if dilate_k > 1:
            mask = dilate(mask, dilate_k)
    recon_blend = blend * recon + (1.0 - blend) * corrupt
    repaired = corrupt * (1.0 - mask) + recon_blend * mask
    return corrupt, repaired, mask


def corruption_weight(
    score_map: torch.Tensor,
    min_weight: float,
) -> torch.Tensor:
    weight = 1.0 - score_map.mean(dim=(1, 2, 3))
    return weight.clamp(min=min_weight)


def train_classifier(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"classifier epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")


def train_classifier_weighted(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y, w in loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="none")
            loss = (loss * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"classifier epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")


def adaloss_from_residual(residual: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    a = alpha.clone()
    a = torch.where(a.abs() < eps, a + eps * torch.sign(a + 1e-9), a)
    a = torch.where((a - 2.0).abs() < eps, a - eps, a)
    b = (a - 2.0).abs().clamp_min(eps)
    return (b / a) * ((residual.pow(2) / b + 1.0).pow(a / 2.0) - 1.0)


def alpha_from_score(score: torch.Tensor, alpha_min: float, alpha_max: float) -> torch.Tensor:
    a = alpha_min + (alpha_max - alpha_min) * score
    return torch.clamp(a, alpha_min, alpha_max)


def train_classifier_adaloss(
    loader: DataLoader,
    model: torch.nn.Module,
    unet: TinyUNet2D,
    alpha_head: AlphaHead2D,
    device: torch.device,
    rng: torch.Generator,
    modes: Tuple[str, ...],
    epochs: int,
    lr: float,
    alpha_min: float,
    alpha_max: float,
    oracle: bool,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            corrupt, mask_gt = corrupt_batch(x, rng, modes=modes)
            with torch.no_grad():
                recon = unet(corrupt)
                resid_img = (recon - corrupt).pow(2).mean(dim=1, keepdim=True)
                if oracle:
                    score = mask_gt
                else:
                    logits = alpha_head(resid_img)
                    score = torch.sigmoid(logits)
                score_mean = score.mean(dim=(1, 2, 3))
            alpha = alpha_from_score(score_mean, alpha_min, alpha_max)
            logits = model(corrupt)
            ce = F.cross_entropy(logits, y, reduction="none")
            loss = adaloss_from_residual(ce, alpha).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"classifier epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")


def eval_classifier(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return float(correct / max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha-guided repair improves robustness (CIFAR-10).")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--weights-dir", type=str, default="weights_twophase")
    parser.add_argument("--train-modes", type=str, default="occlusion,blur")
    parser.add_argument("--eval-unseen", type=str, default="saltpepper,copy")
    parser.add_argument("--mask-topk", type=float, default=0.05)
    parser.add_argument("--dilate-k", type=int, default=1)
    parser.add_argument("--blend", type=float, default=0.3)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--mix-train", action="store_true")
    parser.add_argument("--gate2-mode", type=str, default="repair", choices=["repair", "downweight", "adaloss"])
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--alpha-min", type=float, default=-10.0)
    parser.add_argument("--alpha-max", type=float, default=1.9)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tf)

    idx = torch.randperm(len(train_set))[: args.max_samples]
    subset = Subset(train_set, idx.tolist())
    train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    unet = TinyUNet2D().to(device)
    alpha_head = AlphaHead2D(in_ch=1).to(device)
    unet.load_state_dict(torch.load(os.path.join(args.weights_dir, "unet.pth"), map_location=device))
    alpha_head.load_state_dict(torch.load(os.path.join(args.weights_dir, "alpha_head.pth"), map_location=device))
    unet.eval()
    alpha_head.eval()

    rng = torch.Generator(device=device).manual_seed(args.seed + 11)
    train_modes = tuple(m.strip() for m in args.train_modes.split(",") if m.strip())
    unseen_modes = tuple(m.strip() for m in args.eval_unseen.split(",") if m.strip())

    # Dataset A: clean
    clean_loader = train_loader

    # Dataset B/C: corrupted or repaired
    def make_repair_loader(use_repair: bool, oracle: bool = False):
        xs = []
        ys = []
        coverages = []
        for x, y in train_loader:
            x = x.to(device)
            if use_repair:
                corrupt, repaired, mask = repair_batch(
                    x,
                    unet,
                    alpha_head,
                    rng,
                    train_modes,
                    args.mask_topk,
                    args.dilate_k,
                    oracle=oracle,
                    blend=args.blend,
                )
                x_out = repaired
            else:
                corrupt, _ = corrupt_batch(x, rng, modes=train_modes)
                x_out = corrupt
            xs.append(x_out.cpu())
            ys.append(y)
            if use_repair:
                coverages.append(mask.mean().item())
        return DataLoader(
            list(zip(torch.cat(xs, dim=0), torch.cat(ys, dim=0))),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        ), (float(np.mean(coverages)) if coverages else 0.0)

    def make_weighted_loader(oracle: bool = False):
        xs = []
        ys = []
        ws = []
        for x, y in train_loader:
            x = x.to(device)
            corrupt, mask_gt = corrupt_batch(x, rng, modes=train_modes)
            with torch.no_grad():
                recon = unet(corrupt)
                resid = (recon - corrupt).pow(2).mean(dim=1, keepdim=True)
                if oracle:
                    score = mask_gt
                else:
                    logits = alpha_head(resid)
                    score = torch.sigmoid(logits)
            w = corruption_weight(score, args.min_weight)
            xs.append(corrupt.cpu())
            ys.append(y)
            ws.append(w.cpu())
        return DataLoader(
            list(zip(torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(ws, dim=0))),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

    corrupt_loader, _ = make_repair_loader(use_repair=False)
    repair_loader, repair_cov = make_repair_loader(use_repair=True, oracle=args.oracle)
    weighted_loader = make_weighted_loader(oracle=args.oracle)

    if args.mix_train:
        mix_x = []
        mix_y = []
        for x, y in train_loader:
            x = x.to(device)
            corrupt, repaired, _ = repair_batch(
                x,
                unet,
                alpha_head,
                rng,
                train_modes,
                args.mask_topk,
                args.dilate_k,
                oracle=args.oracle,
                blend=args.blend,
            )
            mix_x.extend([x.cpu(), corrupt.cpu(), repaired.cpu()])
            mix_y.extend([y, y, y])
        mix_loader = DataLoader(
            list(zip(torch.cat(mix_x, dim=0), torch.cat(mix_y, dim=0))),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    else:
        mix_loader = None

    # Train three classifiers
    clf_clean = make_resnet18().to(device)
    clf_corrupt = make_resnet18().to(device)
    clf_repair = make_resnet18().to(device)
    clf_mix = make_resnet18().to(device) if mix_loader is not None else None

    print("\n=== Train A: clean ===")
    train_classifier(clean_loader, clf_clean, device, args.epochs, args.lr)
    print("\n=== Train B: corrupted ===")
    train_classifier(corrupt_loader, clf_corrupt, device, args.epochs, args.lr)
    if args.gate2_mode == "repair":
        print("\n=== Train C: alpha-repaired ===")
        train_classifier(repair_loader, clf_repair, device, args.epochs, args.lr)
        print(f"avg repair coverage: {repair_cov * 100:.2f}%")
    elif args.gate2_mode == "downweight":
        print("\n=== Train C: alpha-weighted ===")
        train_classifier_weighted(weighted_loader, clf_repair, device, args.epochs, args.lr)
    else:
        print("\n=== Train C: AdaLoss (alpha-modulated CE) ===")
        train_classifier_adaloss(
            train_loader,
            clf_repair,
            unet,
            alpha_head,
            device,
            rng,
            train_modes,
            args.epochs,
            args.lr,
            args.alpha_min,
            args.alpha_max,
            args.oracle,
        )
    if mix_loader is not None:
        print("\n=== Train D: mixed (clean+corrupt+repair) ===")
        train_classifier(mix_loader, clf_mix, device, args.epochs, args.lr)

    # Eval: clean test
    acc_clean_a = eval_classifier(test_loader, clf_clean, device)
    acc_clean_b = eval_classifier(test_loader, clf_corrupt, device)
    acc_clean_c = eval_classifier(test_loader, clf_repair, device)

    # Eval: corrupted test (seen)
    def corrupt_eval_loader(modes: Tuple[str, ...]):
        xs = []
        ys = []
        for x, y in test_loader:
            x = x.to(device)
            corrupt, _ = corrupt_batch(x, rng, modes=modes)
            xs.append(corrupt.cpu())
            ys.append(y)
        return DataLoader(
            list(zip(torch.cat(xs, dim=0), torch.cat(ys, dim=0))),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    seen_loader = corrupt_eval_loader(train_modes)
    unseen_loader = corrupt_eval_loader(unseen_modes)

    acc_seen_a = eval_classifier(seen_loader, clf_clean, device)
    acc_seen_b = eval_classifier(seen_loader, clf_corrupt, device)
    acc_seen_c = eval_classifier(seen_loader, clf_repair, device)

    acc_unseen_a = eval_classifier(unseen_loader, clf_clean, device)
    acc_unseen_b = eval_classifier(unseen_loader, clf_corrupt, device)
    acc_unseen_c = eval_classifier(unseen_loader, clf_repair, device)
    acc_unseen_d = eval_classifier(unseen_loader, clf_mix, device) if clf_mix else None

    print("\n=== Results (accuracy) ===")
    print("Train \\ Test | Clean | Corrupt Seen | Corrupt Unseen")
    print(f"Clean train  | {acc_clean_a:.4f} | {acc_seen_a:.4f} | {acc_unseen_a:.4f}")
    print(f"Corrupt train| {acc_clean_b:.4f} | {acc_seen_b:.4f} | {acc_unseen_b:.4f}")
    if args.gate2_mode == "repair":
        label = "Alpha-repair"
    elif args.gate2_mode == "downweight":
        label = "Alpha-weight"
    else:
        label = "AdaLoss"
    print(f"{label} | {acc_clean_c:.4f} | {acc_seen_c:.4f} | {acc_unseen_c:.4f}")
    if clf_mix is not None:
        acc_clean_d = eval_classifier(test_loader, clf_mix, device)
        acc_seen_d = eval_classifier(seen_loader, clf_mix, device)
        print(f"Mix train    | {acc_clean_d:.4f} | {acc_seen_d:.4f} | {acc_unseen_d:.4f}")


if __name__ == "__main__":
    main()
