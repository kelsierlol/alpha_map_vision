import torch
import torch.nn as nn
import torch.nn.functional as F


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
