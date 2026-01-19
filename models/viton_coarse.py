# models/viton_coarse.py
import torch
import torch.nn as nn
from ._base import UNetGenerator
from .warper_tps import ThinPlateWarper

class VITONCoarse(nn.Module):
    def __init__(self, agnostic_c):
        super().__init__()
        self.generator = UNetGenerator(agnostic_c+3)   # agnostic + garment
        self.warper = ThinPlateWarper(in_c=agnostic_c)
    def forward(self, agnostic, garment):
        warped = self.warper(garment, agnostic)
        x = torch.cat([agnostic, warped], 1)
        return self.generator(x), warped
