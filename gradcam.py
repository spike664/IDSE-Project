"""Step 6: Grad-CAM for the trained ResNet-18 regression model.

Run from the solar-soiling folder (venv active):
    python gradcam.py                       # grouped checkpoint, default settings
    python gradcam.py --strategy random

Method: forward pass -> scalar predicted L; backpropagate that scalar to the last
residual stage (layer4, 7x7 feature maps); channel weights = mean gradient per channel;
heatmap = ReLU(sum_k weight_k * feature_map_k), upsampled to 224x224 and overlaid.
The heatmap shows regions that pushed the predicted L UP. It is an interpretation aid;
it does not prove those regions physically caused the power loss.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from data import DATA_ROOT, MEAN, STD, get_transform
from train import build_model


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        pred = self.model(x).squeeze()          # scalar predicted L (regression output)
        pred.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)         # [1, C, 1, 1]
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))  # [1, 1, 7, 7]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(pred.clamp(0, 1))


def load_image(fname, tf):
    img = Image.open(DATA_ROOT / fname).convert("RGB")
    return tf(img)


def to_display(x):
    m = torch.tensor(MEAN).view(3, 1, 1); s = torch.tensor(STD).view(3, 1, 1)
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def pick(df, n, seed=0):
    return df.sample(min(n, len(df)), random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="grouped", choices=["grouped", "random"])
    ap.add_argument("--n", type=int, default=4, help="examples per category")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(f"outputs/checkpoints/resnet18_{a.strategy}_best.pt")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
    model.to(device).eval()
    cam = GradCAM(model, model.layer4)
    tf = get_transform(train=False)

    df = pd.read_csv(f"outputs/preds/resnet18_{a.strategy}_test.csv")
    df["err"] = (df["L_pred"] - df["L_true"]).abs()
    good = df[df["err"] < 0.05]
    cats = {
        "clean_correct": pick(good[good["L_true"] < 0.05], a.n),
        "moderate_correct": pick(good[(good["L_true"] >= 0.25) & (good["L_true"] <= 0.45)], a.n),
        "heavy_correct": pick(good[good["L_true"] >= 0.6], a.n),
        # worst predictions, one per distinct label value (consecutive frames share labels)
        "worst_failures": df.assign(k=df["L_true"].round(3)).sort_values("err", ascending=False)
                            .drop_duplicates("k").head(a.n),
    }
    out = Path(f"outputs/gradcam_{a.strategy}")
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for name, sub in cats.items():
        k = len(sub)
        if k == 0:
            print("no examples for", name); continue
        fig, axes = plt.subplots(2, k, figsize=(3.2 * k, 6.6), squeeze=False)
        for j, (_, r) in enumerate(sub.iterrows()):
            x = load_image(r["filename"], tf).unsqueeze(0).to(device).requires_grad_(True)
            heat, pred = cam(x)
            img = to_display(x[0].detach().cpu())
            axes[0, j].imshow(img); axes[0, j].axis("off")
            axes[0, j].set_title(f"true {r['L_true']:.2f} | pred {pred:.2f}", fontsize=9)
            axes[1, j].imshow(img); axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
            axes[1, j].axis("off")
            log.append({"category": name, "filename": r["filename"],
                        "L_true": r["L_true"], "L_pred": pred})
        plt.suptitle(f"Grad-CAM ({a.strategy} model, layer4) - {name}", fontsize=11)
        plt.tight_layout(); plt.savefig(out / f"{name}.png", dpi=120); plt.close()
        print("saved", out / f"{name}.png")
    pd.DataFrame(log).to_csv(out / "examples_used.csv", index=False)
    print("Examples listed in", out / "examples_used.csv")


if __name__ == "__main__":
    main()
