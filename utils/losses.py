import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class GANLoss(nn.Module):
    """Hinge loss for GAN training."""
    def __init__(self, real: bool):
        super().__init__()
        self.real = real
    
    def forward(self, logits):
        if self.real:
            return torch.relu(1.0 - logits).mean()
        else:
            return torch.relu(1.0 + logits).mean()


class Perceptual(nn.Module):
    """VGG-16 relu_2_2 L2 loss for perceptual similarity."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(weights="DEFAULT").features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
    
    def forward(self, x, y):
        """
        Compute perceptual loss.
        Args:
            x: Predicted image, expected in [-1, 1]
            y: Target image, expected in [-1, 1]
        """
        # Normalize to [0, 1] for VGG
        x_norm = (x + 1) / 2.
        y_norm = (y + 1) / 2.
        return (self.vgg(x_norm) - self.vgg(y_norm)).pow(2).mean()


class VGGPerceptualLoss(Perceptual):
    """Alias for Perceptual loss."""
    pass


# Functional versions
ssim = SSIM(data_range=2.0)
l1 = nn.L1Loss()
l2 = nn.MSELoss()


def total_variation_loss(x):
    """Total variation loss for smoothness."""
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w
