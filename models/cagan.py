import torch
import torch.nn as nn
from ._base import UNetGen, conv

class CAGAN(nn.Module):
    def __init__(self, agnostic_c):
        super().__init__()
        in_c = agnostic_c + 3          # person (agnostic) + garment
        self.G = UNetGen(in_c, 3)
        self.D = nn.Sequential(conv(in_c+3,64,s=2), conv(64,128,s=2),
                               conv(128,256,s=2), nn.Conv2d(256,1,4,1,1))
    def forward(self, a, g): return self.G(torch.cat([a,g],1))
