# utils/pose.py
"""
Pose estimation utilities for Virtual Try-On.
Provides keypoint detection using MediaPipe or fallback methods.
"""
import torch
import numpy as np
from PIL import Image

# Try to import MediaPipe for pose detection
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass


def extract_keypoints(img_rgb, size, use_mediapipe=True):
    """
    Extract pose keypoints and create heatmaps.

    Args:
        img_rgb: Input image as numpy array (H, W, 3) or PIL Image or torch tensor
        size: Output size for heatmaps
        use_mediapipe: Whether to use MediaPipe (if available)

    Returns:
        torch.Tensor: 18-channel pose heatmaps (18 x size x size)
    """
    h = w = size

    # Convert input to numpy array if needed
    if isinstance(img_rgb, torch.Tensor):
        if img_rgb.dim() == 3:
            img_rgb = img_rgb.permute(1, 2, 0).cpu().numpy()
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
    elif isinstance(img_rgb, Image.Image):
        img_rgb = np.array(img_rgb)

    if MEDIAPIPE_AVAILABLE and use_mediapipe:
        return _extract_with_mediapipe(img_rgb, size)
    else:
        return _extract_fallback(img_rgb, size)


def _extract_with_mediapipe(img_rgb, size):
    """
    Extract pose using MediaPipe Pose.
    MediaPipe provides 33 landmarks, we map to 18 (OpenPose format).
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    h = w = size
    heatmaps = np.zeros((18, h, w), dtype=np.float32)

    # Ensure image is uint8
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).astype(np.uint8) if img_rgb.max() <= 1 else img_rgb.astype(np.uint8)

    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # MediaPipe to OpenPose mapping (approximate)
        # OpenPose: 0-nose, 1-neck, 2-rshoulder, 3-relbow, 4-rwrist, 5-lshoulder,
        #           6-lelbow, 7-lwrist, 8-rhip, 9-rknee, 10-rankle, 11-lhip,
        #           12-lknee, 13-lankle, 14-reye, 15-leye, 16-rear, 17-lear
        mp_to_openpose = {
            0: 0,   # nose
            11: 5,  # left shoulder
            12: 2,  # right shoulder
            13: 6,  # left elbow
            14: 3,  # right elbow
            15: 7,  # left wrist
            16: 4,  # right wrist
            23: 11, # left hip
            24: 8,  # right hip
            25: 12, # left knee
            26: 9,  # right knee
            27: 13, # left ankle
            28: 10, # right ankle
        }

        landmarks = results.pose_landmarks.landmark

        for mp_idx, op_idx in mp_to_openpose.items():
            if mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                if lm.visibility > 0.5:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    # Create Gaussian heatmap around keypoint
                    heatmaps[op_idx] = _create_gaussian_heatmap(h, w, y, x)

        # Estimate neck position (between shoulders)
        if 11 < len(landmarks) and 12 < len(landmarks):
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            if l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5:
                neck_x = int((l_shoulder.x + r_shoulder.x) / 2 * w)
                neck_y = int((l_shoulder.y + r_shoulder.y) / 2 * h)
                heatmaps[1] = _create_gaussian_heatmap(h, w, neck_y, neck_x)

    pose.close()
    return torch.from_numpy(heatmaps)


def _extract_fallback(img_rgb, size):
    """
    Fallback pose estimation using simple heuristics.
    Creates approximate pose heatmaps based on image regions.
    """
    h = w = size
    heatmaps = np.zeros((18, h, w), dtype=np.float32)

    # Approximate body part locations (normalized)
    # These are rough estimates for a front-facing person
    body_parts = {
        0: (0.5, 0.12),   # nose
        1: (0.5, 0.22),   # neck
        2: (0.35, 0.25),  # right shoulder
        3: (0.25, 0.40),  # right elbow
        4: (0.20, 0.55),  # right wrist
        5: (0.65, 0.25),  # left shoulder
        6: (0.75, 0.40),  # left elbow
        7: (0.80, 0.55),  # left wrist
        8: (0.40, 0.52),  # right hip
        9: (0.40, 0.72),  # right knee
        10: (0.40, 0.92), # right ankle
        11: (0.60, 0.52), # left hip
        12: (0.60, 0.72), # left knee
        13: (0.60, 0.92), # left ankle
        14: (0.45, 0.10), # right eye
        15: (0.55, 0.10), # left eye
        16: (0.40, 0.12), # right ear
        17: (0.60, 0.12), # left ear
    }

    for idx, (rel_x, rel_y) in body_parts.items():
        x = int(rel_x * w)
        y = int(rel_y * h)
        heatmaps[idx] = _create_gaussian_heatmap(h, w, y, x, sigma=8)

    return torch.from_numpy(heatmaps)


def _create_gaussian_heatmap(h, w, center_y, center_x, sigma=7):
    """Create a Gaussian heatmap centered at (center_y, center_x)."""
    y = np.arange(h)
    x = np.arange(w)
    xx, yy = np.meshgrid(x, y)

    # Ensure center is within bounds
    center_x = max(0, min(w - 1, center_x))
    center_y = max(0, min(h - 1, center_y))

    heatmap = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def draw_pose(img, keypoints, threshold=0.1):
    """
    Draw pose keypoints on image for visualization.

    Args:
        img: PIL Image or numpy array
        keypoints: 18-channel heatmaps
        threshold: Minimum value to consider a keypoint valid

    Returns:
        PIL Image with keypoints drawn
    """
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.copy()

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()

    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85)
    ]

    for i, heatmap in enumerate(keypoints):
        # Find maximum position
        max_val = heatmap.max()
        if max_val > threshold:
            y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            r = 4
            draw.ellipse([x-r, y-r, x+r, y+r], fill=colors[i % len(colors)])

    return img


if __name__ == '__main__':
    # Test pose extraction
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")

    # Create test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    heatmaps = extract_keypoints(test_img, 256)
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Heatmaps range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")
