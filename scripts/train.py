import yaml, argparse, torch, pathlib, sys

# Add project root to path for proper imports
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader
from data.dataset import VITONPairSet
from utils.losses import l1, Perceptual
from models import prgan, cagan, crn, viton

def build(name, ch):
    return {"prgan":prgan.PRGAN, "cagan":cagan.CAGAN,
            "crn":crn.CRN, "viton":viton.VITON}[name](ch)

def main():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    parser = argparse.ArgumentParser(); parser.add_argument("--model",required=True)
    args   = parser.parse_args()

    # Resolve paths relative to project root
    data_root = PROJECT_ROOT / cfg["data_root"]
    save_root = PROJECT_ROOT / cfg["save_root"]

    ds = VITONPairSet(str(data_root), "train", cfg["image_size"])
    dl = DataLoader(ds,batch_size=cfg["batch_size"],shuffle=True,
                    num_workers=cfg["num_workers"],pin_memory=True)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    net = build(args.model, ds.agnostic_channels).to(device)
    if args.model=="viton":
        opt = torch.optim.AdamW([
            {"params": net.coarse.parameters(), "lr": cfg["models"]["viton"]["lr_coarse"]},
            {"params": net.refine.parameters(), "lr": cfg["models"]["viton"]["lr_refine"]}
        ])
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["models"][args.model]["lr"])

    percep = Perceptual().to(device)

    epochs = cfg["models"][args.model]["epochs"]
    for ep in range(1, epochs+1):
        net.train()
        for a,g,t,mask in dl:
            a,g,t,mask = [x.to(device) for x in (a,g,t,mask)]
            if args.model=="viton":
                coarse, out = net(a,g,mask)
                loss = l1(out,t) + .1*percep(out,t)
            else:
                out = net(a,g)
                loss = l1(out,t) + .1*percep(out,t)

            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {ep}/{epochs} | L: {loss.item():.3f}")
        save_root.mkdir(exist_ok=True)
        torch.save(net.state_dict(), save_root / f"{args.model}_{ep:03d}.pth")

if __name__=="__main__":
    torch.backends.cudnn.benchmark = True
    main()
