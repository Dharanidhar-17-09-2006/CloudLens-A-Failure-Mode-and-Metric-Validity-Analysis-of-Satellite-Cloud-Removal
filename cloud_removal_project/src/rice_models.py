"""Larger inpainting generator + PatchGAN discriminator (64x64 tiles)."""
import torch
import torch.nn as nn
from unet_inpaint import DoubleConv


class BigUNetInpainter(nn.Module):
    """4-level U-Net (about 9x the parameters of SmallUNetInpainter at base=48). Same interface."""
    def __init__(self, in_channels=4, out_channels=3, base=48):
        super().__init__()
        b = base
        self.inc = DoubleConv(in_channels, b)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, 2 * b))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(2 * b, 4 * b))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(4 * b, 8 * b))
        self.bott = nn.Sequential(nn.MaxPool2d(2), DoubleConv(8 * b, 16 * b))
        self.u1 = nn.ConvTranspose2d(16 * b, 8 * b, 2, 2); self.c1 = DoubleConv(16 * b, 8 * b)
        self.u2 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2);  self.c2 = DoubleConv(8 * b, 4 * b)
        self.u3 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2);  self.c3 = DoubleConv(4 * b, 2 * b)
        self.u4 = nn.ConvTranspose2d(2 * b, b, 2, 2);      self.c4 = DoubleConv(2 * b, b)
        self.outc = nn.Sequential(nn.Conv2d(b, out_channels, 1), nn.Sigmoid())

    def forward(self, x_masked, mask):
        x1 = self.inc(torch.cat([x_masked, mask], 1))
        x2 = self.d1(x1); x3 = self.d2(x2); x4 = self.d3(x3); xb = self.bott(x4)
        d = self.c1(torch.cat([self.u1(xb), x4], 1))
        d = self.c2(torch.cat([self.u2(d), x3], 1))
        d = self.c3(torch.cat([self.u3(d), x2], 1))
        d = self.c4(torch.cat([self.u4(d), x1], 1))
        pred = self.outc(d)
        return x_masked * (1 - mask) + pred * mask


class PatchDiscriminator(nn.Module):
    """PatchGAN with spectral norm; sees image + mask."""
    def __init__(self, in_ch=4, base=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch, base, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(base, 2 * base, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(2 * base, 4 * base, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(4 * base, 4 * base, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(4 * base, 1, 3, 1, 1)))

    def forward(self, img, mask):
        return self.net(torch.cat([img, mask], 1))


if __name__ == "__main__":
    g = BigUNetInpainter(); x = torch.rand(2, 3, 64, 64); m = (torch.rand(2, 1, 64, 64) > 0.5).float()
    print(g(x, m).shape, sum(p.numel() for p in g.parameters()) / 1e6, "M params")
    print(PatchDiscriminator()(x, m).shape)
