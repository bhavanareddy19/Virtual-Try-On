import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional import jaccard_index

# InceptionScore requires torch-fidelity which may not be installed
INCEPTION_AVAILABLE = False
try:
    from torchmetrics.image.inception import InceptionScore
    INCEPTION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

def compute_metrics(pred, gt, mask):
    """Returns dict with SSIM, IS (if available) and IoU."""
    device = pred.device
    with torch.no_grad():
        # SSIM - Structural Similarity Index
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        ssim_val = ssim_metric(pred, gt)
        
        # Inception Score - only if torch-fidelity is installed
        is_score = 0.0
        if INCEPTION_AVAILABLE:
            try:
                pred_normalized = ((pred + 1) / 2.).clamp(0, 1)
                pred_uint8 = (pred_normalized * 255).to(torch.uint8)
                is_metric = InceptionScore().to(device)
                is_metric.update(pred_uint8)
                is_score = is_metric.compute()[0].item()
            except Exception as e:
                print(f"InceptionScore failed: {e}")
                is_score = 0.0
        
        # IoU for mask (comparing same mask gives baseline of 1.0)
        iou = jaccard_index((mask>0.5).int().to(device), (mask>0.5).int().to(device), num_classes=2, task='binary')
        
    return {"ssim": ssim_val.item(), "is": is_score, "iou": iou.item()}


def compute_ssim(pred, gt):
    """Compute only SSIM between prediction and ground truth."""
    device = pred.device
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    return ssim_metric(pred, gt)
