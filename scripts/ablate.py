"""
Disable refinement or warper to reproduce ablation study for VITON.
"""
import argparse, torch, yaml, pathlib, sys

# Add project root to path for proper imports
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import VITONPairSet
from models import viton as viton_module

def build(name, ch):
    from models import prgan, cagan, crn, viton
    return {"prgan":prgan.PRGAN,"cagan":cagan.CAGAN,
            "crn":crn.CRN,"viton":viton.VITON}[name](ch)

def main():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_refine", action="store_true")
    parser.add_argument("--disable_warper", action="store_true")
    args = parser.parse_args()

    # Resolve paths relative to project root
    data_root = PROJECT_ROOT / cfg["data_root"]

    ds = VITONPairSet(str(data_root), "test", cfg["image_size"])
    viton = build("viton", ds.agnostic_channels)
    if args.disable_refine:  viton.refine = torch.nn.Identity()
    if args.disable_warper:  viton.coarse.warper = torch.nn.Identity()

    print(f"Ablation configuration:")
    print(f"  disable_refine: {args.disable_refine}")
    print(f"  disable_warper: {args.disable_warper}")
    print(f"Model ready for evaluation.")
    # now reuse evaluate code…

if __name__ == "__main__":
    main()
