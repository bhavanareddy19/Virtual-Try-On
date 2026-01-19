import yaml, argparse, torch, json, pathlib, sys

# Add project root to path for proper imports
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import VITONPairSet
from utils.metrics import compute_metrics
from models import prgan, cagan, crn, viton

def build(name,ch):
    return {"prgan":prgan.PRGAN,"cagan":cagan.CAGAN,
            "crn":crn.CRN,"viton":viton.VITON}[name](ch)

def main():
    cfg   = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    parser= argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt",  required=True)
    args  = parser.parse_args()

    # Resolve paths relative to project root
    data_root = PROJECT_ROOT / cfg["data_root"]

    ds = VITONPairSet(str(data_root), "test", cfg["image_size"])
    dl = DataLoader(ds,batch_size=1,shuffle=False)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    net = build(args.model, ds.agnostic_channels).to(device).eval()
    net.load_state_dict(torch.load(args.ckpt, map_location=device))

    scores = []
    with torch.no_grad():
        for a,g,t,mask in tqdm(dl):
            a,g,t,mask = [x.to(device) for x in (a,g,t,mask)]
            if args.model=="viton":
                _, out = net(a,g,mask)
            else:
                out = net(a,g)
            scores.append(compute_metrics(out, t, mask))
    print({k: sum(d[k] for d in scores)/len(scores) for k in scores[0]})

if __name__=="__main__":
    main()
