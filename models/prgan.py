import torch
import torch.nn as nn
from ._base import UNetGen, conv

class PRGenerator(nn.Module):
    def __init__(self, in_c): super().__init__(); self.unet = UNetGen(in_c+3)  # agnostic + garment
    def forward(self,a,g): return self.unet(torch.cat([a,g],1))

class PatchDis(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_c, 64, s=2), conv(64,128,s=2), conv(128,256,s=2),
            nn.Conv2d(256,1,3,1,1))
    def forward(self,x): return self.net(x)

class PRGAN(nn.Module):
    """Wrap generator & discriminator so train.py can access gen/dis."""
    def __init__(self, agnostic_c):
        super().__init__()
        self.gen = PRGenerator(agnostic_c)
        self.dis = PatchDis(3+agnostic_c)
    def forward(self, a, g): return self.gen(a,g)
