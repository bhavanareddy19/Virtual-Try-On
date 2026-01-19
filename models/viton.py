# models/viton.py
import torch.nn as nn
class VITON(nn.Module):
    def __init__(self, agnostic_c):
        super().__init__()
        from .viton_coarse import VITONCoarse
        from .viton_refine import VITONRefine
        self.coarse = VITONCoarse(agnostic_c)
        self.refine = VITONRefine(in_c=3+3+1)  # coarse, warped, mask
    def forward(self, agnostic, garment, vis_mask):
        coarse, warped = self.coarse(agnostic, garment)
        refined = self.refine(coarse, warped, vis_mask)
        return coarse, refined
