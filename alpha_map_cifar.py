import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyUNet2D(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        m = self.mid(d2)
        u2 = self.up2(m)
        c2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.out(c1))


class AlphaController2D(nn.Module):
    def __init__(self, alpha_min: float = -1.5, alpha_max: float = 1.9, alpha0: float = -0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha0 = alpha0

    def forward(self, resid_mag: torch.Tensor, redundancy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([resid_mag, redundancy], dim=1)
        a = self.net(x) + self.alpha0
        return torch.clamp(a, self.alpha_min, self.alpha_max)


def local_redundancy(x: torch.Tensor, window: int) -> torch.Tensor:
    pad = window // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    mean = F.avg_pool2d(x_pad, kernel_size=window, stride=1)
    mean2 = F.avg_pool2d(x_pad * x_pad, kernel_size=window, stride=1)
    var = (mean2 - mean * mean).clamp_min(0.0)
    vmin = var.amin(dim=(2, 3), keepdim=True)
    vmax = var.amax(dim=(2, 3), keepdim=True)
    norm = (var - vmin) / (vmax - vmin + 1e-6)
    redundancy = 1.0 - norm
    return redundancy


def apply_blur_patch(x: torch.Tensor, top: int, left: int, size: int) -> torch.Tensor:
    kernel = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=x.dtype, device=x.device
    )
    kernel = kernel / kernel.sum()
    k = kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    blurred = F.conv2d(x, k, padding=1, groups=x.size(1))
    x[:, :, top:top + size, left:left + size] = blurred[:, :, top:top + size, left:left + size]
    return x


def corrupt_batch(
    x: torch.Tensor,
    rng: torch.Generator,
    occlusion_size: int = 10,
    blur_size: int = 8,
    sp_size: int = 6,
    copy_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, _, h, w = x.shape
    mask = torch.zeros((bsz, 1, h, w), device=x.device)
    corrupted = x.clone()

    for i in range(bsz):
        # Occlusion square
        top = torch.randint(0, h - occlusion_size, (1,), generator=rng).item()
        left = torch.randint(0, w - occlusion_size, (1,), generator=rng).item()
        corrupted[i, :, top:top + occlusion_size, left:left + occlusion_size] = 0.0
        mask[i, 0, top:top + occlusion_size, left:left + occlusion_size] = 1.0

        # Blur patch
        top = torch.randint(0, h - blur_size, (1,), generator=rng).item()
        left = torch.randint(0, w - blur_size, (1,), generator=rng).item()
        corrupted[i:i + 1] = apply_blur_patch(corrupted[i:i + 1], top, left, blur_size)
        mask[i, 0, top:top + blur_size, left:left + blur_size] = 1.0

        # Salt & pepper patch
        top = torch.randint(0, h - sp_size, (1,), generator=rng).item()
        left = torch.randint(0, w - sp_size, (1,), generator=rng).item()
        sp = torch.rand((sp_size, sp_size), generator=rng, device=x.device)
        sp = (sp > 0.5).float()
        corrupted[i, :, top:top + sp_size, left:left + sp_size] = sp.unsqueeze(0).repeat(3, 1, 1)
        mask[i, 0, top:top + sp_size, left:left + sp_size] = 1.0

        # Copy-paste patch
        src_top = torch.randint(0, h - copy_size, (1,), generator=rng).item()
        src_left = torch.randint(0, w - copy_size, (1,), generator=rng).item()
        dst_top = torch.randint(0, h - copy_size, (1,), generator=rng).item()
        dst_left = torch.randint(0, w - copy_size, (1,), generator=rng).item()
        corrupted[i, :, dst_top:dst_top + copy_size, dst_left:dst_left + copy_size] = \
            corrupted[i, :, src_top:src_top + copy_size, src_left:src_left + copy_size]
        mask[i, 0, dst_top:dst_top + copy_size, dst_left:dst_left + copy_size] = 1.0

    return corrupted, mask


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(labels_sorted.sum(), 1e-12)
    return np.trapezoid(precision, recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha map demo on CIFAR-10 with known corruption masks.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-unet", type=int, default=3)
    parser.add_argument("--epochs-alpha", type=int, default=3)
    parser.add_argument("--lr-unet", type=float, default=1e-3)
    parser.add_argument("--lr-alpha", type=float, default=5e-4)
    parser.add_argument("--redundancy-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)

    idx = torch.randperm(len(train_set))[: args.max_samples]
    subset = Subset(train_set, idx.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TinyUNet2D().to(device)
    controller = AlphaController2D().to(device)

    # Stage 1: Train UNet reconstruction
    opt_unet = torch.optim.Adam(model.parameters(), lr=args.lr_unet)
    model.train()
    for epoch in range(1, args.epochs_unet + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            loss = F.l1_loss(recon, x)
            opt_unet.zero_grad(set_to_none=True)
            loss.backward()
            opt_unet.step()
            total += loss.item()
        print(f"unet epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")

    # Stage 2: Train alpha controller on known corruptions
    opt_alpha = torch.optim.Adam(controller.parameters(), lr=args.lr_alpha)
    controller.train()
    model.eval()
    rng = torch.Generator(device=device).manual_seed(args.seed + 1)

    for epoch in range(1, args.epochs_alpha + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            corrupt, mask = corrupt_batch(x, rng)
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
        print(f"alpha epoch {epoch:02d} | loss {total / max(1, len(loader)):.4f}")

    # Quick evaluation on a few batches
    controller.eval()
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= 10:
                break
            x = x.to(device)
            corrupt, mask = corrupt_batch(x, rng)
            recon = model(corrupt)
            resid = (recon - corrupt).abs().mean(dim=1, keepdim=True)
            redundancy = local_redundancy(corrupt, args.redundancy_window).mean(dim=1, keepdim=True)
            alpha = controller(resid, redundancy)
            score = (-(alpha - controller.alpha0)).cpu().numpy().reshape(-1)
            scores.append(score)
            labels.append(mask.cpu().numpy().reshape(-1))
    scores_np = np.concatenate(scores)
    labels_np = np.concatenate(labels)
    print(f"PR-AUC (quick): {pr_auc(scores_np, labels_np):.4f}")

    if args.plot:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x, _ = next(iter(loader))
        x = x.to(device)
        corrupt, mask = corrupt_batch(x, rng)
        with torch.no_grad():
            recon = model(corrupt)
            resid = (recon - corrupt).abs().mean(dim=1, keepdim=True)
            redundancy = local_redundancy(corrupt, args.redundancy_window).mean(dim=1, keepdim=True)
            alpha = controller(resid, redundancy)

        os.makedirs(args.output_dir, exist_ok=True)
        img = corrupt[0].permute(1, 2, 0).cpu().numpy()
        heat = alpha[0, 0].cpu().numpy()
        m = mask[0, 0].cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img)
        axs[0].set_title("Corrupted")
        axs[0].axis("off")
        axs[1].imshow(heat, cmap="viridis")
        axs[1].set_title("Alpha map")
        axs[1].axis("off")
        axs[2].imshow(m, cmap="Reds")
        axs[2].set_title("Corruption mask")
        axs[2].axis("off")
        fig.tight_layout()
        out = os.path.join(args.output_dir, "cifar_alpha_demo.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
