# data/dataset.py
"""
Dataset module for Virtual Try-On
Supports VITON-HD format and custom data formats
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json


class VITONPairSet(Dataset):
    """
    Virtual Try-On Dataset for paired training.

    Expected directory structure:
    data_root/
    ├── train/
    │   ├── image/              # Person images
    │   ├── cloth/              # Garment images
    │   ├── cloth-mask/         # Garment masks (optional)
    │   ├── agnostic-v3.2/      # Agnostic person images (person without clothes)
    │   ├── image-parse-v3/     # Segmentation maps (optional)
    │   └── pairs.txt           # Pairs file (person_img cloth_img per line)
    └── test/
        └── (same structure)

    For simpler setups without agnostic images:
    - If agnostic images don't exist, we create them from person images
    """

    def __init__(self, data_root, split='train', image_size=256, use_pose=False):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train' or 'test'
            image_size: Target image size (will be resized to image_size x image_size)
            use_pose: Whether to include pose heatmaps in agnostic representation
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.use_pose = use_pose

        # Define paths
        self.split_dir = os.path.join(data_root, split)
        self.image_dir = os.path.join(self.split_dir, 'image')
        self.cloth_dir = os.path.join(self.split_dir, 'cloth')
        self.cloth_mask_dir = os.path.join(self.split_dir, 'cloth-mask')
        self.agnostic_dir = os.path.join(self.split_dir, 'agnostic-v3.2')
        self.parse_dir = os.path.join(self.split_dir, 'image-parse-v3')

        # Load pairs
        self.pairs = self._load_pairs()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Agnostic channels: 3 (RGB) + 18 (pose heatmaps) if use_pose else 3
        self._agnostic_channels = 3 + (18 if use_pose else 0)

    @property
    def agnostic_channels(self):
        """Return number of channels in the agnostic representation."""
        return self._agnostic_channels

    def _load_pairs(self):
        """Load image pairs from pairs.txt or create from directory listing."""
        pairs_file = os.path.join(self.split_dir, 'pairs.txt')

        if os.path.exists(pairs_file):
            pairs = []
            with open(pairs_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            pairs.append((parts[0], parts[1]))
            return pairs

        # Fallback: Create pairs from directory listing
        if os.path.exists(self.image_dir) and os.path.exists(self.cloth_dir):
            images = sorted([f for f in os.listdir(self.image_dir)
                           if f.endswith(('.jpg', '.png', '.jpeg'))])
            cloths = sorted([f for f in os.listdir(self.cloth_dir)
                           if f.endswith(('.jpg', '.png', '.jpeg'))])

            # Match by name or create sequential pairs
            pairs = []
            for img in images:
                # Try to find matching cloth
                base_name = os.path.splitext(img)[0]
                matching_cloth = None
                for cloth in cloths:
                    if base_name in cloth or cloth.replace('_cloth', '') == img:
                        matching_cloth = cloth
                        break
                if matching_cloth is None and cloths:
                    # Use random/first cloth for demo purposes
                    matching_cloth = cloths[len(pairs) % len(cloths)]
                if matching_cloth:
                    pairs.append((img, matching_cloth))
            return pairs

        # Create demo data if nothing exists
        return self._create_demo_data()

    def _create_demo_data(self):
        """Create demonstration data for testing the pipeline."""
        print(f"No data found in {self.data_root}. Creating demo data...")

        # Create directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.cloth_dir, exist_ok=True)
        os.makedirs(self.agnostic_dir, exist_ok=True)

        # Create synthetic demo images
        demo_pairs = []
        for i in range(4):
            # Create person image (white background with colored rectangle as person)
            person_img = Image.new('RGB', (self.image_size, self.image_size), (240, 240, 240))

            # Create garment image (colored rectangle)
            garment_img = Image.new('RGB', (self.image_size, self.image_size), (200, 200, 200))
            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]

            # Draw simple shapes using PIL
            from PIL import ImageDraw

            # Person - draw body shape
            draw = ImageDraw.Draw(person_img)
            draw.ellipse([80, 20, 176, 80], fill=(200, 170, 150))  # head
            draw.rectangle([70, 80, 186, 200], fill=colors[i])  # torso (wearing clothes)
            draw.rectangle([80, 200, 100, 256], fill=(100, 80, 150))  # left leg
            draw.rectangle([156, 200, 176, 256], fill=(100, 80, 150))  # right leg

            # Garment - flat garment
            draw_g = ImageDraw.Draw(garment_img)
            draw_g.rectangle([40, 40, 216, 180], fill=colors[(i+1)%4])  # t-shirt
            draw_g.rectangle([10, 40, 40, 120], fill=colors[(i+1)%4])  # left sleeve
            draw_g.rectangle([216, 40, 246, 120], fill=colors[(i+1)%4])  # right sleeve

            # Agnostic person (person without the garment area)
            agnostic_img = person_img.copy()
            draw_a = ImageDraw.Draw(agnostic_img)
            draw_a.rectangle([70, 80, 186, 200], fill=(128, 128, 128))  # mask torso

            # Save images
            person_path = f'person_{i:04d}.jpg'
            garment_path = f'garment_{i:04d}.jpg'

            person_img.save(os.path.join(self.image_dir, person_path))
            garment_img.save(os.path.join(self.cloth_dir, garment_path))
            agnostic_img.save(os.path.join(self.agnostic_dir, person_path))

            demo_pairs.append((person_path, garment_path))

        # Save pairs file
        with open(os.path.join(self.split_dir, 'pairs.txt'), 'w') as f:
            for p, g in demo_pairs:
                f.write(f'{p} {g}\n')

        print(f"Created {len(demo_pairs)} demo pairs in {self.split_dir}")
        return demo_pairs

    def _load_image(self, dir_path, filename, transform=None):
        """Load an image from directory."""
        if transform is None:
            transform = self.transform

        path = os.path.join(dir_path, filename)

        # Try different extensions
        if not os.path.exists(path):
            base, ext = os.path.splitext(filename)
            for new_ext in ['.jpg', '.png', '.jpeg']:
                new_path = os.path.join(dir_path, base + new_ext)
                if os.path.exists(new_path):
                    path = new_path
                    break

        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            return transform(img)
        else:
            # Return zeros if image not found
            return torch.zeros(3, self.image_size, self.image_size)

    def _create_agnostic(self, person_img, parse_map=None):
        """
        Create agnostic representation from person image.
        Agnostic = person image with torso/clothes region masked out.
        """
        if parse_map is not None:
            # Use parsing map to mask upper body region
            mask = (parse_map > 0).float()
            agnostic = person_img * (1 - mask) + 0.5 * mask  # Gray fill
        else:
            # Simple center masking as fallback
            agnostic = person_img.clone()
            h, w = self.image_size, self.image_size
            # Mask center region (approximate torso)
            agnostic[:, h//4:3*h//4, w//4:3*w//4] = 0.5

        return agnostic

    def _get_pose_heatmaps(self, img_name):
        """Get pose heatmaps if available."""
        if not self.use_pose:
            return None

        # Try to load precomputed pose
        pose_dir = os.path.join(self.split_dir, 'pose')
        pose_path = os.path.join(pose_dir, img_name.replace('.jpg', '_pose.npy').replace('.png', '_pose.npy'))

        if os.path.exists(pose_path):
            import numpy as np
            pose = np.load(pose_path)
            return torch.from_numpy(pose).float()

        # Return zeros if pose not available
        return torch.zeros(18, self.image_size, self.image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            agnostic: Person image with clothes masked (C x H x W)
            garment: Target garment image (3 x H x W)
            target: Ground truth result - person wearing garment (3 x H x W)
            mask: Visibility mask for the garment region (1 x H x W)
        """
        person_name, garment_name = self.pairs[idx]

        # Load person image (this is also our target for paired training)
        target = self._load_image(self.image_dir, person_name)

        # Load garment
        garment = self._load_image(self.cloth_dir, garment_name)

        # Load or create agnostic representation
        if os.path.exists(self.agnostic_dir):
            agnostic = self._load_image(self.agnostic_dir, person_name)
        else:
            agnostic = self._create_agnostic(target)

        # Add pose heatmaps to agnostic if enabled
        if self.use_pose:
            pose = self._get_pose_heatmaps(person_name)
            if pose is not None:
                agnostic = torch.cat([agnostic, pose], dim=0)

        # Load or create mask
        mask_path = os.path.join(self.cloth_mask_dir, garment_name)
        if os.path.exists(mask_path):
            mask = self._load_image(self.cloth_mask_dir, garment_name, self.mask_transform)
            mask = mask.mean(dim=0, keepdim=True)  # Convert to single channel
        else:
            # Create simple mask from garment (non-background pixels)
            mask = (garment.mean(dim=0, keepdim=True) > -0.9).float()

        return agnostic, garment, target, mask


class VITONInferenceSet(Dataset):
    """
    Dataset for inference - allows mixing person and garment freely.
    """

    def __init__(self, person_images, garment_images, image_size=256):
        """
        Args:
            person_images: List of paths to person images
            garment_images: List of paths to garment images
            image_size: Target image size
        """
        self.person_images = person_images
        self.garment_images = garment_images
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @property
    def agnostic_channels(self):
        return 3

    def __len__(self):
        return len(self.person_images) * len(self.garment_images)

    def __getitem__(self, idx):
        person_idx = idx // len(self.garment_images)
        garment_idx = idx % len(self.garment_images)

        person = Image.open(self.person_images[person_idx]).convert('RGB')
        garment = Image.open(self.garment_images[garment_idx]).convert('RGB')

        person = self.transform(person)
        garment = self.transform(garment)

        # Create simple agnostic (center masked)
        agnostic = person.clone()
        h, w = self.image_size, self.image_size
        agnostic[:, h//4:3*h//4, w//4:3*w//4] = 0.0

        # Simple mask
        mask = torch.ones(1, h, w)
        mask[:, :h//4, :] = 0
        mask[:, 3*h//4:, :] = 0

        return agnostic, garment, person, mask


if __name__ == '__main__':
    # Test the dataset
    print("Testing VITONPairSet...")
    ds = VITONPairSet('./data', 'train', 256)
    print(f"Dataset size: {len(ds)}")
    print(f"Agnostic channels: {ds.agnostic_channels}")

    if len(ds) > 0:
        a, g, t, m = ds[0]
        print(f"Agnostic shape: {a.shape}")
        print(f"Garment shape: {g.shape}")
        print(f"Target shape: {t.shape}")
        print(f"Mask shape: {m.shape}")
