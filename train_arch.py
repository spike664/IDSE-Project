"""Step 9: train.py with an --arch option (resnet18 | efficientnet_b0). train.py itself is untouched.

Step 4: transfer-learning regression for soiling power loss (L, fraction 0-1).

Run from the solar-soiling folder (venv active):
    python train_arch.py --strategy grouped --arch efficientnet_b0 [--seed 1|2]
    python train_arch.py --strategy grouped --arch resnet18        # identical to train.py
Fallback only if B0 does not train: add --lr-ft 3e-5 (the tag gets _lr3e-05, report it as tuned)
Quick debug run (a few minutes):
    python train.py --strategy grouped --limit 2000 --epochs 2 --head-epochs 1

Phase A: backbone frozen, train the new 1-output head (--head-epochs).
Phase B: unfreeze everything, fine-tune with a lower LR; early stopping on val MSE.
The splits come from splits/splits.csv and are never changed here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from data import SoilingDataset

BINS = [(0.0, 0.15, "low"), (0.15, 0.5, "mid"), (0.5, 1.01, "high")]


ARCHS = ("resnet18", "efficientnet_b0")


def build_model(arch="resnet18", pretrained=True):
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 1)       # regression head -> predicted L
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        # classifier = Sequential(Dropout(0.2), Linear(1280, 1000)); keep the dropout
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(arch)
    return m


def head_of(model, arch):
    return model.fc if arch == "resnet18" else model.classifier


def metrics(y, p):
    err = y - p
    out = {"MAE": float(np.abs(err).mean()),
           "RMSE": float(np.sqrt((err ** 2).mean())),
           "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    for lo, hi, name in BINS:
        m = (y >= lo) & (y < hi)
        out[f"MAE_{name}"] = float(np.abs(err[m]).mean()) if m.any() else float("nan")
    return out


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ys, ps, names = [], [], []
    for x, y, n in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            p = model(x).squeeze(1).float()
        ys.append(y.numpy()); ps.append(p.clamp(0, 1).cpu().numpy()); names += list(n)
    return np.concatenate(ys), np.concatenate(ps), names


def run_epoch(model, loader, opt, device, amp):
    model.train()
    tot, n = 0.0, 0
    lossf = nn.MSELoss()
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            p = model(x).squeeze(1)
        loss = lossf(p.float(), y)
        loss.backward()
        opt.step()
        tot += loss.item() * len(y); n += len(y)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["random", "grouped"], required=True)
    ap.add_argument("--arch", choices=ARCHS, default="resnet18")
    ap.add_argument("--epochs", type=int, default=12, help="max fine-tune epochs (phase B)")
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: images per split")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop", action="store_true",
                    help="warp each frame so the panel fills it (panel_crop.py)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    tag = (f"{a.arch}_{a.strategy}" + ("_crop" if a.crop else "")
           + (f"_seed{a.seed}" if a.seed != 42 else "")
           + (f"_lr{a.lr_ft:g}" if a.lr_ft != 1e-4 else "")
           + (f"_lim{a.limit}" if a.limit else ""))
    for d in ("checkpoints", "preds", "plots"):
        Path(f"outputs/{d}").mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | AMP(bf16): {amp} | run: {tag}")

    dss = {s: SoilingDataset(a.strategy, s, crop=a.crop) for s in ("train", "val", "test")}
    if a.limit:
        for d in dss.values():
            d.df = d.df.sample(min(a.limit, len(d.df)), random_state=0).reset_index(drop=True)
    loaders = {s: DataLoader(d, batch_size=a.batch_size, shuffle=(s == "train"),
                             num_workers=a.workers, pin_memory=amp,
                             persistent_workers=a.workers > 0)
               for s, d in dss.items()}
    print({s: len(d) for s, d in dss.items()})

    model = build_model(a.arch, not a.no_pretrained).to(device)
    head = head_of(model, a.arch)
    best, bad, hist = 1e9, 0, []
    ckpt = Path(f"outputs/checkpoints/{tag}_best.pt")
    t_start = time.time()

    def validate():
        y, p, _ = predict(model, loaders["val"], device, amp)
        return float(((y - p) ** 2).mean()), float(np.abs(y - p).mean())

    def save_if_best(vmse):
        nonlocal best
        if vmse < best:
            best = vmse
            torch.save({"state_dict": model.state_dict(), "args": vars(a)}, ckpt)
            return True
        return False

    # ---- Phase A: head only
    for p_ in model.parameters():
        p_.requires_grad = False
    for p_ in head.parameters():
        p_.requires_grad = True
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr_head)
    for ep in range(1, a.head_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], opt, device, amp)
        vmse, vmae = validate()
        imp = save_if_best(vmse)
        hist.append({"phase": "head", "epoch": ep, "train_mse": tr, "val_mse": vmse, "val_mae": vmae})
        print(f"[head {ep}/{a.head_epochs}] train MSE {tr:.4f} | val MSE {vmse:.4f} "
              f"MAE {vmae:.4f} | {time.time()-t0:.0f}s {'*best*' if imp else ''}")

    # ---- Phase B: full fine-tune
    for p_ in model.parameters():
        p_.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr_ft, weight_decay=1e-4)
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], opt, device, amp)
        vmse, vmae = validate()
        imp = save_if_best(vmse)
        bad = 0 if imp else bad + 1
        hist.append({"phase": "finetune", "epoch": ep, "train_mse": tr, "val_mse": vmse, "val_mae": vmae})
        print(f"[ft {ep}/{a.epochs}] train MSE {tr:.4f} | val MSE {vmse:.4f} "
              f"MAE {vmae:.4f} | {time.time()-t0:.0f}s {'*best*' if imp else ''}")
        if bad >= a.patience:
            print("Early stopping."); break
    train_min = (time.time() - t_start) / 60

    h = pd.DataFrame(hist)
    h.to_csv(f"outputs/{tag}_history.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(h["train_mse"].values, label="train MSE"); plt.plot(h["val_mse"].values, label="val MSE")
    plt.axvline(a.head_epochs - 0.5, ls="--", c="gray")
    plt.xlabel("epoch (dashed: head-only -> fine-tune)"); plt.ylabel("MSE"); plt.legend()
    plt.title(tag); plt.tight_layout(); plt.savefig(f"outputs/plots/{tag}_curves.png", dpi=120)

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    rows = []
    for s in ("val", "test"):
        y, p, names = predict(model, loaders[s], device, amp)
        pd.DataFrame({"filename": names, "L_true": y, "L_pred": p}).to_csv(
            f"outputs/preds/{tag}_{s}.csv", index=False)
        m = metrics(y, p)
        rows.append({"run": tag, "set": s, "n": len(y), **m})
        print(f"\n{tag} | {s}: MAE {m['MAE']:.4f} ({m['MAE']*100:.2f} pts) | "
              f"RMSE {m['RMSE']:.4f} | R2 {m['R2']:+.4f} | "
              f"MAE low/mid/high {m['MAE_low']:.3f}/{m['MAE_mid']:.3f}/{m['MAE_high']:.3f}")
    res = pd.DataFrame(rows).round(4)
    res["train_minutes"] = round(train_min, 1)
    res["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    out = Path("outputs/results_step9.csv")   # separate file: results.csv stays untouched
    res.to_csv(out, mode="a", header=not out.exists(), index=False)
    print("\nAppended to outputs/results_step9.csv | training time: %.1f min" % train_min)


if __name__ == "__main__":
    main()
