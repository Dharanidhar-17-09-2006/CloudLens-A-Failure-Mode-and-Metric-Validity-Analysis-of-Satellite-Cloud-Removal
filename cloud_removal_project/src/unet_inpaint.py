"""
Phase 3: Lightweight U-Net Inpainting Architecture.
Optimized for 64x64 satellite tiles.
Takes 4 channels (RGB + Mask) and predicts 3 RGB channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class SmallUNetInpainter(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 4, base_channels * 8)
        )
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_channels * 2, base_channels)
        
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid() # normalized [0, 1]
        )

    def forward(self, x_masked, mask):
        """
        x_masked: (B, 3, H, W) normalized to [0, 1]
        mask: (B, 1, H, W) binary (1 for cloud, 0 for clear)
        """
        inp = torch.cat([x_masked, mask], dim=1)
        
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        xb = self.bottleneck(x3)
        
        u1 = self.up1(xb)
        u1 = torch.cat([u1, x3], dim=1)
        d1 = self.conv_up1(u1)
        
        u2 = self.up2(d1)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.conv_up2(u2)
        
        u3 = self.up3(d2)
        u3 = torch.cat([u3, x1], dim=1)
        d3 = self.conv_up3(u3)
        
        pred = self.outc(d3)
        
        # Composite: retain known clean pixels exactly, only fill cloud region
        out = x_masked * (1.0 - mask) + pred * mask
        return out

if __name__ == "__main__":
    model = SmallUNetInpainter()
    x = torch.rand(2, 3, 64, 64)
    m = torch.randint(0, 2, (2, 1, 64, 64)).float()
    out = model(x, m)
    print("SmallUNetInpainter test pass, output shape:", out.shape)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
