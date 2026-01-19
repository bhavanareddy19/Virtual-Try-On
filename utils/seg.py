# utils/seg.py
"""
Semantic segmentation utilities for Virtual Try-On.
Provides human parsing/segmentation using torchvision models or simple methods.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Try to import segmentation model
SEGMENTATION_MODEL = None
try:
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    TORCHVISION_SEG_AVAILABLE = True
except ImportError:
    TORCHVISION_SEG_AVAILABLE = False


def human_parse(img_rgb, size, use_deep_learning=True, device='cpu'):
    """
    Perform human parsing/segmentation.

    Args:
        img_rgb: Input image as numpy array (H, W, 3), PIL Image, or torch tensor
        size: Output size
        use_deep_learning: Whether to use deep learning model (if available)
        device: Device for inference

    Returns:
        torch.Tensor: Segmentation mask (1 x size x size) with values in [0, 1]
    """
    # Convert input to PIL Image
    if isinstance(img_rgb, torch.Tensor):
        if img_rgb.dim() == 3:
            img_rgb = img_rgb.permute(1, 2, 0).cpu().numpy()
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        img_rgb = Image.fromarray(img_rgb)
    elif isinstance(img_rgb, np.ndarray):
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        img_rgb = Image.fromarray(img_rgb)

    if TORCHVISION_SEG_AVAILABLE and use_deep_learning:
        return _segment_with_deeplabv3(img_rgb, size, device)
    else:
        return _segment_simple(img_rgb, size)


def _get_segmentation_model(device='cpu'):
    """Load and cache the segmentation model."""
    global SEGMENTATION_MODEL
    if SEGMENTATION_MODEL is None and TORCHVISION_SEG_AVAILABLE:
        SEGMENTATION_MODEL = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        SEGMENTATION_MODEL.eval()
        SEGMENTATION_MODEL.to(device)
    return SEGMENTATION_MODEL


def _segment_with_deeplabv3(img_pil, size, device='cpu'):
    """
    Segment image using DeepLabV3.
    COCO classes include person (class 15).
    """
    from torchvision import transforms

    model = _get_segmentation_model(device)
    if model is None:
        return _segment_simple(img_pil, size)

    # Prepare input
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']

    # Get person class (class 15 in COCO)
    pred = output.argmax(1)
    person_mask = (pred == 15).float()

    # Resize to target size if needed
    if person_mask.shape[-1] != size:
        person_mask = F.interpolate(person_mask.unsqueeze(0), size=(size, size), mode='nearest').squeeze(0)

    return person_mask.cpu()


def _segment_simple(img_pil, size):
    """
    Simple segmentation using color-based thresholding.
    Assumes person is in center and background is relatively uniform.
    """
    img_array = np.array(img_pil.resize((size, size)))

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    # Simple thresholding - assume person is different from background
    # Use edges and center prior
    h, w = size, size

    # Create center prior (assume person is in center)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    center_prior = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * (h // 3) ** 2))

    # Compute local variance as texture indicator
    from scipy import ndimage
    variance = ndimage.generic_filter(gray, np.var, size=5)
    variance_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)

    # Combine center prior with variance
    mask = (center_prior > 0.3) & (variance_norm > 0.1)
    mask = mask.astype(np.float32)

    # Morphological operations to clean up
    try:
        from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
        mask = binary_fill_holes(mask)
        mask = binary_dilation(mask, iterations=2)
        mask = binary_erosion(mask, iterations=1)
    except:
        pass

    return torch.from_numpy(mask).unsqueeze(0).float()


def get_clothing_mask(segmentation, clothing_classes=None):
    """
    Extract clothing region from full segmentation.

    Args:
        segmentation: Full body segmentation
        clothing_classes: List of class IDs for clothing (model-specific)

    Returns:
        Binary mask for clothing region
    """
    if clothing_classes is None:
        # Default: assume upper body region (rough approximation)
        h, w = segmentation.shape[-2:]
        mask = torch.zeros_like(segmentation)
        # Upper body region (neck to waist)
        mask[..., int(h * 0.15):int(h * 0.55), :] = segmentation[..., int(h * 0.15):int(h * 0.55), :]
        return mask
    else:
        # Use specific clothing classes
        clothing_mask = torch.zeros_like(segmentation)
        for cls in clothing_classes:
            clothing_mask = clothing_mask | (segmentation == cls)
        return clothing_mask.float()


def create_agnostic_mask(person_image, segmentation=None, size=256, device='cpu'):
    """
    Create an agnostic representation by masking the clothing region.

    Args:
        person_image: Person image tensor (C x H x W) or (B x C x H x W)
        segmentation: Optional pre-computed segmentation
        size: Image size
        device: Device

    Returns:
        Agnostic image with clothing region masked
    """
    if segmentation is None:
        segmentation = human_parse(person_image, size, device=device)

    # Create clothing mask (upper body area)
    h, w = size, size
    clothing_region = torch.zeros(1, h, w)
    clothing_region[:, int(h * 0.15):int(h * 0.60), int(w * 0.25):int(w * 0.75)] = 1.0

    # Combine with person segmentation
    mask = segmentation * clothing_region

    # Apply mask to create agnostic
    if person_image.dim() == 3:
        person_image = person_image.unsqueeze(0)

    agnostic = person_image.clone()
    mask = mask.unsqueeze(0) if mask.dim() == 3 else mask

    # Fill masked region with gray
    agnostic = agnostic * (1 - mask) + 0.5 * mask

    return agnostic.squeeze(0), mask.squeeze(0)


def visualize_segmentation(image, segmentation, alpha=0.5):
    """
    Overlay segmentation on image for visualization.

    Args:
        image: Original image (PIL Image or tensor)
        segmentation: Segmentation mask
        alpha: Overlay transparency

    Returns:
        PIL Image with segmentation overlay
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        if image.min() < 0:
            image = (image + 1) / 2
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if segmentation.ndim == 3:
        segmentation = segmentation[0]

    # Create colored overlay
    overlay = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    overlay[segmentation > 0.5] = [0, 255, 0]  # Green for person

    overlay_img = Image.fromarray(overlay)
    overlay_img = overlay_img.resize(image.size)

    # Blend
    result = Image.blend(image.convert('RGB'), overlay_img, alpha)
    return result


if __name__ == '__main__':
    # Test segmentation
    print(f"TorchVision segmentation available: {TORCHVISION_SEG_AVAILABLE}")

    # Create test image
    test_img = Image.new('RGB', (256, 256), (200, 200, 200))

    mask = human_parse(test_img, 256)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
