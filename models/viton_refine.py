# models/viton_refine.py
import torch
import torch.nn as nn
from ._base import UNetGenerator

class VITONRefine(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.unet = UNetGenerator(in_c)
    def forward(self, coarse_out, warped, vis_mask):
        x = torch.cat([coarse_out, warped, vis_mask], 1)
        return self.unet(x)
