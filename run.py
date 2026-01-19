#!/usr/bin/env python3
"""
Simple entry point for Deep Virtual Try-On.

Usage:
    python run.py                           # Quick test
    python run.py train prgan               # Train PRGAN
    python run.py inference viton           # Run inference with VITON
    python run.py evaluate prgan checkpoint/prgan_020.pth  # Evaluate
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def print_help():
    print("""
Deep Virtual Try-On System
==========================

Commands:
  python run.py test                    Test all models
  python run.py demo [model]            Run inference demo (models: prgan, cagan, crn, viton)
  python run.py train <model>           Train a model (e.g., python run.py train prgan)
  python run.py evaluate <model> <ckpt> Evaluate a model
  
Examples:
  python run.py test                    # Test all models load correctly
  python run.py demo prgan              # Run demo with PRGAN
  python run.py demo viton              # Run demo with VITON
  python run.py train prgan             # Train PRGAN for 20 epochs
  python run.py train viton             # Train VITON for 20 epochs
  python run.py evaluate prgan checkpoints/prgan_020.pth
  
For more options, run: python demo.py --help
""")


def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'test':
        subprocess.run([sys.executable, 'demo.py', '--test'], cwd=PROJECT_ROOT)
    
    elif cmd == 'demo':
        model = sys.argv[2] if len(sys.argv) > 2 else 'prgan'
        subprocess.run([sys.executable, 'demo.py', '--model', model], cwd=PROJECT_ROOT)
    
    elif cmd == 'train':
        if len(sys.argv) < 3:
            print("Usage: python run.py train <model>")
            print("Models: prgan, cagan, crn, viton")
            return
        model = sys.argv[2]
        subprocess.run([sys.executable, 'scripts/train.py', '--model', model], cwd=PROJECT_ROOT)
    
    elif cmd == 'evaluate':
        if len(sys.argv) < 4:
            print("Usage: python run.py evaluate <model> <checkpoint>")
            return
        model = sys.argv[2]
        checkpoint = sys.argv[3]
        subprocess.run([sys.executable, 'scripts/evaluate.py', '--model', model, '--ckpt', checkpoint], cwd=PROJECT_ROOT)
    
    elif cmd in ['help', '-h', '--help']:
        print_help()
    
    else:
        print(f"Unknown command: {cmd}")
        print_help()


if __name__ == '__main__':
    main()
