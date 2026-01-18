import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.up(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64): # Increased base from 32 to 64
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base, num_groups=min(4, base))
        self.down1 = Downsample(base, base * 2)
        self.enc2 = ConvBlock(base * 2, base * 2, num_groups=min(8, base * 2))
        self.down2 = Downsample(base * 2, base * 4)
        self.enc3 = ConvBlock(base * 4, base * 4, num_groups=min(8, base * 4))
        self.down3 = Downsample(base * 4, base * 8)
        self.enc4 = ConvBlock(base * 8, base * 8, num_groups=min(8, base * 8))
        self.down4 = Downsample(base * 8, base * 16)

        # Mid
        self.mid = ConvBlock(base * 16, base * 16, num_groups=min(8, base * 16))

        # Decoder
        self.up4 = Upsample(base * 16, base * 8)
        self.dec4 = ConvBlock(base * 16, base * 8, num_groups=min(8, base * 8))
        self.up3 = Upsample(base * 8, base * 4)
        self.dec3 = ConvBlock(base * 8, base * 4, num_groups=min(8, base * 4))
        self.up2 = Upsample(base * 4, base * 2)
        self.dec2 = ConvBlock(base * 4, base * 2, num_groups=min(8, base * 2))
        self.up1 = Upsample(base * 2, base)
        self.dec1 = ConvBlock(base * 2, base, num_groups=min(4, base))

        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)
        d3 = self.down3(e3)
        e4 = self.enc4(d3)
        d4 = self.down4(e4)

        # Mid path
        m = self.mid(d4)

        # Decoder path
        u4 = self.up4(m)
        c4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(c4)
        c3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.out(c1))

