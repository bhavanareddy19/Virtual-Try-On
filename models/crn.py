import torch.nn as nn, torch
from ._base import conv

class _Refine(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_c,64), conv(64,64), nn.Conv2d(64,out_c,3,1,1), nn.Tanh())
    def forward(self,x): return self.net(x)

class CRN(nn.Module):
    """Cascaded refinement ‑ 3 stages 64→128→256."""
    def __init__(self, agnostic_c):
        super().__init__()
        self.s64  = _Refine(agnostic_c+3,  3)
        self.s128 = _Refine(agnostic_c+6,  3)
        self.s256 = _Refine(agnostic_c+6,  3)

    def forward(self, a, g):
        # Stage 1: 64x64
        a64 = nn.functional.interpolate(a, 64, mode='bilinear', align_corners=False)
        g64 = nn.functional.interpolate(g, 64, mode='bilinear', align_corners=False)
        x64 = torch.cat([a64, g64], 1)
        out1 = self.s64(x64)

        # Stage 2: 128x128
        a128 = nn.functional.interpolate(a, 128, mode='bilinear', align_corners=False)
        g128 = nn.functional.interpolate(g, 128, mode='bilinear', align_corners=False)
        out1_128 = nn.functional.interpolate(out1, 128, mode='bilinear', align_corners=False)
        x128 = torch.cat([a128, g128, out1_128], 1)
        out2 = self.s128(x128)

        # Stage 3: 256x256
        out2_256 = nn.functional.interpolate(out2, 256, mode='bilinear', align_corners=False)
        x256 = torch.cat([a, g, out2_256], 1)
        out3 = self.s256(x256)
        return out3
