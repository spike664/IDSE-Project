"""Step 7: robustness and background-reliance tests on the grouped TEST set.

Run from the solar-soiling folder (venv active):
    python robustness.py                  # full grouped test set (~5-8 min)
    python robustness.py --limit 500      # quick debug run

The trained model is NOT retrained or tuned. We perturb the test images and measure how
the predictions change:
  * photometric: brightness, contrast, Gaussian noise, blur, JPEG compression
  * occlusion: grey out the bottom strip (panel base/stand) and the top-right corner,
    the two regions Grad-CAM highlighted on clean panels, to test background reliance.
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import DATA_ROOT, MEAN, STD
from train import build_model

MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)


class Raw(Dataset):
    """Resized (224) uint8 image, no normalisation, no augmentation."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tf = T.Compose([T.Resize((224, 224)), T.PILToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(DATA_ROOT / r["filename"]).convert("RGB")), float(r["L"])


# ---- perturbations on float tensors in [0,1], shape [B,3,H,W]
def brightness(f):
    return lambda x, g: (x * f).clamp(0, 1)

def contrast(f):
    def fn(x, g):
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - m) * f + m).clamp(0, 1)
    return fn

def noise(s):
    return lambda x, g: (x + torch.randn(x.shape, generator=g, device=x.device) * s).clamp(0, 1)

def blur(sigma):
    k = int(2 * round(3 * sigma) + 1)
    return lambda x, g: TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])

def jpeg(q):
    def fn(x, g):
        out = []
        for im in (x * 255).round().byte().cpu():
            buf = io.BytesIO()
            Image.fromarray(im.permute(1, 2, 0).numpy()).save(buf, "JPEG", quality=q)
            buf.seek(0)
            out.append(torch.from_numpy(np.array(Image.open(buf).convert("RGB"))).permute(2, 0, 1))
        return torch.stack(out).float().div(255).to(x.device)
    return fn

def occlude(bottom=False, topright=False):
    def fn(x, g):
        x = x.clone()
        if bottom:
            x[:, :, 179:, :] = 0.5            # bottom 20%: panel base / stand
        if topright:
            x[:, :, :56, 168:] = 0.5           # top-right corner (neighbouring panel)
        return x
    return fn


def box(y0, y1, x0, x1):
    def fn(x, g):
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = 0.5
        return x
    return fn


# Controls for the occlusion test: same-size grey boxes in other places, to tell a real
# dependence on a region apart from the grey box itself confusing the model.
CONTROLS = [
    ("clean (no change)", lambda x, g: x),
    ("control: top-left corner 56x56", box(0, 56, 0, 56)),
    ("control: bottom-left corner 56x56", box(168, 224, 0, 56)),
    ("control: top-centre 56x56", box(0, 56, 84, 140)),
    ("occlude panel centre 56x56", box(84, 140, 84, 140)),
]

CONDITIONS = [
    ("clean (no change)", lambda x, g: x),
    ("brightness x0.6", brightness(0.6)), ("brightness x0.8", brightness(0.8)),
    ("brightness x1.2", brightness(1.2)), ("brightness x1.4", brightness(1.4)),
    ("contrast x0.6", contrast(0.6)), ("contrast x1.4", contrast(1.4)),
    ("noise sigma 0.02", noise(0.02)), ("noise sigma 0.05", noise(0.05)),
    ("noise sigma 0.10", noise(0.10)),
    ("blur sigma 1", blur(1.0)), ("blur sigma 2", blur(2.0)),
    ("JPEG q50", jpeg(50)), ("JPEG q20", jpeg(20)), ("JPEG q10", jpeg(10)),
    ("occlude bottom 20%", occlude(bottom=True)),
    ("occlude top-right corner", occlude(topright=True)),
    ("occlude both regions", occlude(bottom=True, topright=True)),
]


@torch.no_grad()
def predict(model, X, fn, device, g, bs=128):
    preds = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device).float() / 255.0
        x = fn(x, g)
        x = (x - MEAN_T.to(device)) / STD_T.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            p = model(x).squeeze(1).float()
        preds.append(p.clamp(0, 1).cpu())
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="grouped", choices=["grouped", "random"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--controls-only", action="store_true",
                    help="only the clean baseline plus the occlusion control boxes")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    ck = Path(f"outputs/checkpoints/resnet18_{a.strategy}_best.pt")
    model.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
    model.to(device).eval()

    df = pd.read_csv("splits/splits.csv")
    df = df[df[f"{a.strategy}_split"] == "test"]
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0)
    print(f"Loading {len(df)} test images into memory ...")
    xs, ys = [], []
    for x, y in DataLoader(Raw(df), batch_size=64, num_workers=0):
        xs.append(x); ys.append(y)
    X, y = torch.cat(xs), torch.cat(ys).numpy()

    g = torch.Generator(device=device); g.manual_seed(42)
    clean_pred, rows = None, []
    conds = CONTROLS if a.controls_only else CONDITIONS
    for name, fn in conds:
        g.manual_seed(42)
        p = predict(model, X, fn, device, g)
        if clean_pred is None:
            clean_pred = p
        e = y - p
        rows.append({"condition": name,
                     "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                     "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                     "bias(pred-true)": (p - y).mean(),
                     "mean|pred shift|": np.abs(p - clean_pred).mean()})
        r = rows[-1]
        print(f"{name:26s} MAE {r['MAE']:.4f} | RMSE {r['RMSE']:.4f} | R2 {r['R2']:+.4f} | "
              f"bias {r['bias(pred-true)']:+.4f} | shift {r['mean|pred shift|']:.4f}")

    res = pd.DataFrame(rows)
    res["dMAE_vs_clean"] = res["MAE"] - res.loc[0, "MAE"]
    Path("outputs/eval").mkdir(parents=True, exist_ok=True)
    out = (f"outputs/eval/robustness_{a.strategy}_test"
           + ("_controls" if a.controls_only else "")
           + (f"_lim{a.limit}" if a.limit else ""))
    res.round(4).to_csv(out + ".csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 5))
    plt.barh(res["condition"][::-1], res["MAE"][::-1])
    plt.axvline(res.loc[0, "MAE"], color="r", ls="--", lw=1, label="clean MAE")
    plt.xlabel("MAE (fraction of L)"); plt.legend()
    plt.title(f"Robustness - {a.strategy} test set"); plt.tight_layout()
    plt.savefig(out + ".png", dpi=120)
    print("\nSaved", out + ".csv and .png")


if __name__ == "__main__":
    main()
