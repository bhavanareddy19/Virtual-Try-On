# models/warper_tps.py
"""
Thin Plate Spline (TPS) warping module for garment deformation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinPlateWarper(nn.Module):
    """
    Thin Plate Spline warping for garment transformation.

    Uses a regressor network to predict control point offsets,
    then applies TPS transformation to warp the garment.
    """
    def __init__(self, in_c, num_ctrl=9):
        """
        Args:
            in_c: Number of input channels for the regressor
            num_ctrl: Number of control points (must be a perfect square, e.g., 9=3x3)
        """
        super().__init__()
        self.num_ctrl = num_ctrl
        self.grid_size = int(num_ctrl ** 0.5)

        # Regressor network to predict control point offsets
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_ctrl * 2)  # 2D offsets for each control point
        )

        # Initialize to small offsets
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, garment, feat):
        """
        Warp garment based on features.

        Args:
            garment: Garment image tensor (B, C, H, W)
            feat: Feature tensor from which to predict deformation (B, in_c, H', W')

        Returns:
            Warped garment tensor (B, C, H, W)
        """
        B, C, H, W = garment.shape
        device = garment.device

        # Predict offsets
        offsets = self.regressor(feat).view(B, self.num_ctrl, 2)
        offsets = torch.tanh(offsets) * 0.3  # Limit offset magnitude

        # Create source control points (regular grid)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.grid_size, device=device),
            torch.linspace(-1, 1, self.grid_size, device=device),
            indexing='ij'
        )
        src_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (num_ctrl, 2)
        src_pts = src_pts.unsqueeze(0).repeat(B, 1, 1)  # (B, num_ctrl, 2)

        # Destination points = source + predicted offsets
        dst_pts = src_pts + offsets

        # Apply TPS transformation
        warped = self._tps_warp(garment, src_pts, dst_pts)

        return warped

    def _tps_warp(self, img, src_pts, dst_pts):
        """
        Apply TPS warping using custom implementation.

        This is a simplified TPS that uses a grid-based approach
        for compatibility with different PyTorch/Kornia versions.
        """
        B, C, H, W = img.shape
        device = img.device

        # Try to use Kornia's TPS if available and working
        try:
            return self._tps_warp_kornia(img, src_pts, dst_pts)
        except Exception:
            # Fallback to simplified warping
            return self._tps_warp_simple(img, src_pts, dst_pts)

    def _tps_warp_kornia(self, img, src_pts, dst_pts):
        """Use Kornia's TPS warping."""
        import kornia
        from kornia.geometry.transform import get_tps_transform, warp_image_tps

        B, C, H, W = img.shape

        # Kornia's TPS expects points in different format in different versions
        # Try the newer API first
        try:
            # Newer Kornia API
            kernel_weights, affine_weights = get_tps_transform(dst_pts, src_pts)
            warped = warp_image_tps(img, src_pts, kernel_weights, affine_weights, (H, W))
        except TypeError:
            # Older API or different signature
            try:
                weights = get_tps_transform(dst_pts, src_pts)
                if isinstance(weights, tuple):
                    kernel_weights, affine_weights = weights
                    warped = warp_image_tps(img, src_pts, kernel_weights, affine_weights, (H, W))
                else:
                    warped = warp_image_tps(img, src_pts, dst_pts, weights, (H, W))
            except:
                raise RuntimeError("Kornia TPS API incompatible")

        return warped

    def _tps_warp_simple(self, img, src_pts, dst_pts):
        """
        Simplified warping using bilinear interpolation.
        This is a fallback that approximates TPS with a simpler transformation.
        """
        B, C, H, W = img.shape
        device = img.device

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

        # Compute displacement field using RBF interpolation (simplified)
        # For each pixel, compute weighted sum of control point displacements
        displacements = dst_pts - src_pts  # (B, num_ctrl, 2)

        # Reshape for broadcasting
        grid_flat = grid.view(B, H * W, 2)  # (B, H*W, 2)
        src_pts_expanded = src_pts.unsqueeze(1)  # (B, 1, num_ctrl, 2)
        grid_expanded = grid_flat.unsqueeze(2)  # (B, H*W, 1, 2)

        # Compute distances to each control point
        distances = torch.norm(grid_expanded - src_pts_expanded, dim=-1)  # (B, H*W, num_ctrl)

        # RBF weights (Gaussian)
        sigma = 0.5
        weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))  # (B, H*W, num_ctrl)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize

        # Compute displacement for each pixel
        pixel_displacements = torch.einsum('bnc,bcd->bnd', weights, displacements)  # (B, H*W, 2)
        pixel_displacements = pixel_displacements.view(B, H, W, 2)

        # Apply displacement
        sampling_grid = grid - pixel_displacements  # Inverse mapping

        # Sample from image
        warped = F.grid_sample(img, sampling_grid, mode='bilinear',
                              padding_mode='border', align_corners=True)

        return warped


class AffineWarper(nn.Module):
    """
    Simple affine transformation warper as an alternative to TPS.
    """
    def __init__(self, in_c):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)  # Affine transformation parameters
        )

        # Initialize to identity transform
        nn.init.zeros_(self.regressor[-1].weight)
        self.regressor[-1].bias.data = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)

    def forward(self, garment, feat):
        B = garment.size(0)

        # Predict affine parameters
        theta = self.regressor(feat).view(B, 2, 3)

        # Create grid and sample
        grid = F.affine_grid(theta, garment.size(), align_corners=True)
        warped = F.grid_sample(garment, grid, mode='bilinear',
                              padding_mode='border', align_corners=True)

        return warped


if __name__ == '__main__':
    # Test the warpers
    batch_size = 2
    in_channels = 3
    feat_channels = 64
    H, W = 256, 256

    # Create dummy inputs
    garment = torch.randn(batch_size, in_channels, H, W)
    feat = torch.randn(batch_size, feat_channels, H, W)

    # Test TPS warper
    print("Testing ThinPlateWarper...")
    tps_warper = ThinPlateWarper(feat_channels, num_ctrl=9)
    warped = tps_warper(garment, feat)
    print(f"  Input shape: {garment.shape}")
    print(f"  Output shape: {warped.shape}")

    # Test Affine warper
    print("\nTesting AffineWarper...")
    affine_warper = AffineWarper(feat_channels)
    warped_affine = affine_warper(garment, feat)
    print(f"  Input shape: {garment.shape}")
    print(f"  Output shape: {warped_affine.shape}")

    print("\nAll tests passed!")
