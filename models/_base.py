import torch
import torch.nn as nn

def conv(in_c, out_c, k=3, s=1, p=1, norm=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p)]
    if norm: layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.LeakyReLU(.2, inplace=True))
    return nn.Sequential(*layers)

class ResBlk(nn.Module):
    def __init__(self, c): super().__init__(); self.net = nn.Sequential(conv(c,c), conv(c,c,norm=False))
    def forward(self,x): return x + self.net(x)

class UNetGen(nn.Module):
    """4‑level UNet, output tanh."""
    def __init__(self, in_c, out_c=3, nf=64):
        super().__init__()
        self.d1 = conv(in_c,   nf)
        self.d2 = conv(nf,     nf*2, s=2)
        self.d3 = conv(nf*2,   nf*4, s=2)
        self.d4 = conv(nf*4,   nf*8, s=2)
        self.u3 = nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1)
        self.u2 = nn.ConvTranspose2d(nf*8, nf*2, 4, 2, 1)
        self.u1 = nn.ConvTranspose2d(nf*4, nf,   4, 2, 1)
        self.out= nn.Sequential(nn.Conv2d(nf*2, out_c, 3,1,1), nn.Tanh())
    def forward(self,x):
        d1,d2,d3,d4 = self.d1(x), self.d2(self.d1(x)), self.d3(self.d2(self.d1(x))), self.d4(self.d3(self.d2(self.d1(x))))
        u3 = self.u3(d4)
        u2 = self.u2(torch.cat([u3,d3],1))
        u1 = self.u1(torch.cat([u2,d2],1))
        return self.out(torch.cat([u1,d1],1))

# Alias for backward compatibility
UNetGenerator = UNetGen
