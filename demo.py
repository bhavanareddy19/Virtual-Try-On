#!/usr/bin/env python3
"""
Deep Virtual Try-On Demo Script

This script demonstrates the virtual try-on pipeline using the implemented models.
It can run with demo data or real images.

Usage:
    python demo.py                              # Run with demo data
    python demo.py --model viton               # Use specific model
    python demo.py --person img.jpg --cloth c.jpg  # Use custom images
    python demo.py --train --model prgan --epochs 5  # Quick training demo
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """Load configuration from config.yaml."""
    import yaml
    config_path = PROJECT_ROOT / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        'image_size': 256,
        'batch_size': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_root': './checkpoints',
        'data_root': './data'
    }


def get_model(model_name, agnostic_channels=3):
    """Load the specified model."""
    from models import PRGAN, CAGAN, CRN, VITON

    models = {
        'prgan': PRGAN,
        'cagan': CAGAN,
        'crn': CRN,
        'viton': VITON
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")

    return models[model_name](agnostic_channels)


def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return True
    return False


def preprocess_image(img_path, size=256):
    """Load and preprocess an image."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path).convert('RGB')
    return transform(img)


def postprocess_image(tensor):
    """Convert tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def create_demo_images(size=256):
    """Create simple demo images for testing."""
    from PIL import ImageDraw

    # Create person image
    person = Image.new('RGB', (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(person)

    # Draw person silhouette
    draw.ellipse([size//2-40, 20, size//2+40, 80], fill=(200, 170, 150))  # head
    draw.rectangle([size//2-50, 80, size//2+50, 180], fill=(100, 150, 200))  # torso
    draw.rectangle([size//2-60, 80, size//2-50, 140], fill=(100, 150, 200))  # left arm
    draw.rectangle([size//2+50, 80, size//2+60, 140], fill=(100, 150, 200))  # right arm
    draw.rectangle([size//2-40, 180, size//2-15, size-10], fill=(50, 50, 100))  # left leg
    draw.rectangle([size//2+15, 180, size//2+40, size-10], fill=(50, 50, 100))  # right leg

    # Create garment image (t-shirt)
    garment = Image.new('RGB', (size, size), (220, 220, 220))
    draw_g = ImageDraw.Draw(garment)

    draw_g.rectangle([50, 40, size-50, 160], fill=(255, 100, 100))  # main body
    draw_g.rectangle([20, 40, 50, 100], fill=(255, 100, 100))  # left sleeve
    draw_g.rectangle([size-50, 40, size-20, 100], fill=(255, 100, 100))  # right sleeve
    draw_g.arc([size//2-30, 20, size//2+30, 60], 180, 0, fill=(220, 220, 220), width=15)  # collar

    # Create agnostic (person without clothes)
    agnostic = person.copy()
    draw_a = ImageDraw.Draw(agnostic)
    draw_a.rectangle([size//2-50, 80, size//2+50, 180], fill=(128, 128, 128))  # mask torso

    return person, garment, agnostic


def demo_inference(args):
    """Run inference demo."""
    print("\n" + "="*60)
    print("Deep Virtual Try-On - Inference Demo")
    print("="*60)

    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    size = config['image_size']

    print(f"\nDevice: {device}")
    print(f"Model: {args.model}")
    print(f"Image size: {size}x{size}")

    # Load or create images
    if args.person and args.cloth:
        print(f"\nLoading custom images...")
        person = preprocess_image(args.person, size)
        garment = preprocess_image(args.cloth, size)
        agnostic = person.clone()
        # Simple agnostic: mask center
        agnostic[:, size//4:3*size//4, size//4:3*size//4] = 0.0
    else:
        print(f"\nCreating demo images...")
        person_pil, garment_pil, agnostic_pil = create_demo_images(size)

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        person = transform(person_pil)
        garment = transform(garment_pil)
        agnostic = transform(agnostic_pil)

    # Create batch
    agnostic = agnostic.unsqueeze(0).to(device)
    garment = garment.unsqueeze(0).to(device)
    mask = torch.ones(1, 1, size, size).to(device)

    # Load model
    print(f"\nLoading {args.model} model...")
    model = get_model(args.model, agnostic_channels=3).to(device)

    # Try to load checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_dir = Path(config['save_root'])
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob(f"{args.model}_*.pth"))
            if ckpts:
                ckpt_path = str(sorted(ckpts)[-1])

    if ckpt_path and os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found - using random weights (for demo purposes)")

    # Run inference
    print(f"\nRunning inference...")
    model.eval()
    with torch.no_grad():
        if args.model == 'viton':
            coarse, output = model(agnostic, garment, mask)
        else:
            output = model(agnostic, garment)

    # Save results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # Save input images
    if not (args.person and args.cloth):
        person_pil.save(output_dir / 'input_person.png')
        garment_pil.save(output_dir / 'input_garment.png')
        agnostic_pil.save(output_dir / 'input_agnostic.png')

    # Save output
    result = postprocess_image(output)
    result.save(output_dir / f'output_{args.model}.png')

    if args.model == 'viton':
        coarse_result = postprocess_image(coarse)
        coarse_result.save(output_dir / 'output_coarse.png')

    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"  - output_{args.model}.png")

    # Create comparison grid
    from utils.vis import save_grid
    if not (args.person and args.cloth):
        grid_tensors = torch.cat([
            transform(person_pil).unsqueeze(0),
            transform(garment_pil).unsqueeze(0),
            transform(agnostic_pil).unsqueeze(0),
            output.cpu()
        ], dim=0)
        save_grid(grid_tensors, output_dir / 'comparison_grid.png', nrow=4)
        print(f"  - comparison_grid.png")

    print("\nDemo complete!")


def demo_training(args):
    """Run a quick training demo."""
    print("\n" + "="*60)
    print("Deep Virtual Try-On - Training Demo")
    print("="*60)

    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")

    # Create dataset
    from data.dataset import VITONPairSet
    from torch.utils.data import DataLoader

    print(f"\nCreating dataset...")
    data_root = config['data_root']
    ds = VITONPairSet(data_root, 'train', config['image_size'])
    dl = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True)

    print(f"Dataset size: {len(ds)}")
    print(f"Agnostic channels: {ds.agnostic_channels}")

    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model, ds.agnostic_channels).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Setup training
    from utils.losses import l1, Perceptual
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    percep_loss = Perceptual().to(device)

    # Training loop
    print(f"\nStarting training...")
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for batch_idx, (agnostic, garment, target, mask) in enumerate(dl):
            agnostic = agnostic.to(device)
            garment = garment.to(device)
            target = target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            if args.model == 'viton':
                coarse, output = model(agnostic, garment, mask)
            else:
                output = model(agnostic, garment)

            loss = l1(output, target) + 0.1 * percep_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_dir = Path(config['save_root'])
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model}_demo.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # Save sample output
    model.eval()
    with torch.no_grad():
        if args.model == 'viton':
            _, sample = model(agnostic[:1], garment[:1], mask[:1])
        else:
            sample = model(agnostic[:1], garment[:1])

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    result = postprocess_image(sample)
    result.save(output_dir / f'training_sample_{args.model}.png')
    print(f"Sample saved: {output_dir / f'training_sample_{args.model}.png'}")

    print("\nTraining demo complete!")


def test_models(args):
    """Test all models load correctly."""
    print("\n" + "="*60)
    print("Deep Virtual Try-On - Model Test")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    models_to_test = ['prgan', 'cagan', 'crn', 'viton']
    size = 256
    batch_size = 2

    # Create dummy inputs
    agnostic = torch.randn(batch_size, 3, size, size).to(device)
    garment = torch.randn(batch_size, 3, size, size).to(device)
    mask = torch.ones(batch_size, 1, size, size).to(device)

    print("\nTesting models...")
    for model_name in models_to_test:
        try:
            model = get_model(model_name, agnostic_channels=3).to(device)
            model.eval()

            with torch.no_grad():
                if model_name == 'viton':
                    coarse, output = model(agnostic, garment, mask)
                    print(f"  {model_name}: OK (coarse: {coarse.shape}, output: {output.shape})")
                else:
                    output = model(agnostic, garment)
                    print(f"  {model_name}: OK (output: {output.shape})")

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {num_params:,}")

        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")

    print("\nModel test complete!")


def main():
    parser = argparse.ArgumentParser(description='Deep Virtual Try-On Demo')
    parser.add_argument('--model', type=str, default='prgan',
                       choices=['prgan', 'cagan', 'crn', 'viton'],
                       help='Model to use')
    parser.add_argument('--person', type=str, default=None,
                       help='Path to person image')
    parser.add_argument('--cloth', type=str, default=None,
                       help='Path to garment image')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--train', action='store_true',
                       help='Run training demo')
    parser.add_argument('--test', action='store_true',
                       help='Test all models')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs for demo')

    args = parser.parse_args()

    if args.test:
        test_models(args)
    elif args.train:
        demo_training(args)
    else:
        demo_inference(args)


if __name__ == '__main__':
    main()
