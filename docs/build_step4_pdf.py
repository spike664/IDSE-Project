"""Build docs/Step4_ResNet18_Training.pdf (Step 4 documentation).

Usage (from the project folder, venv active):
    python docs/build_step4_pdf.py

Nothing is retrained here. All numbers come from the files written by train.py
(outputs/results.csv, outputs/*_history.csv, outputs/preds/*.csv, checkpoint args) and from
splits/splits.csv (baseline). Metrics are recomputed from the per-image prediction files and
cross-checked against outputs/results.csv. Raw images are not touched.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG_DIR = ROOT / "docs" / "step4_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF = ROOT / "docs" / "Step4_ResNet18_Training.pdf"
STRATS = ("grouped", "random")
SETS = ("val", "test")
BINS = [(0.0, 0.15, "low", "low (L &lt; 0.15)"), (0.15, 0.5, "mid", "mid (0.15-0.5)"),
        (0.5, 1.01, "high", "high (L &gt;= 0.5)")]


def metrics(y, p):
    e = y - p
    out = {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
           "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}
    for lo, hi, k, _ in BINS:
        m = (y >= lo) & (y < hi)
        out[f"MAE_{k}"] = np.abs(e[m]).mean()
        out[f"bias_{k}"] = (p[m] - y[m]).mean()
        out[f"n_{k}"] = int(m.sum())
    out["bias"] = (p - y).mean()
    out["n"] = len(y)
    return out


# ---------------------------------------------------------------- load run artefacts
res = pd.read_csv(OUT / "results.csv")
hist = {st: pd.read_csv(OUT / f"resnet18_{st}_history.csv") for st in STRATS}
hist_dbg = pd.read_csv(OUT / "resnet18_grouped_lim2000_history.csv")
preds = {(st, sb): pd.read_csv(OUT / "preds" / f"resnet18_{st}_{sb}.csv")
         for st in STRATS for sb in SETS}
ck_args = {}
for st in STRATS:
    ck = torch.load(OUT / "checkpoints" / f"resnet18_{st}_best.pt", map_location="cpu",
                    weights_only=False)
    ck_args[st] = ck["args"]
    n_params = sum(v.numel() for k, v in ck["state_dict"].items()
                   if "running" not in k and "num_batches" not in k)

# ---------------------------------------------------------------- recompute CNN metrics
cnn = {}
for (st, sb), d in preds.items():
    y, p = d["L_true"].to_numpy(float), d["L_pred"].to_numpy(float)
    cnn[(st, sb)] = metrics(y, p)
    cnn[(st, sb)]["clamped0"] = float((p <= 0).mean())
    cnn[(st, sb)]["clamped1"] = float((p >= 1).mean())


def res_row(st, sb):
    return res[(res.run == f"resnet18_{st}") & (res.set == sb)].iloc[-1]


csv_ok = all(np.allclose([cnn[k][m] for m in ("MAE", "RMSE", "R2", "MAE_low", "MAE_mid", "MAE_high")],
                         res_row(*k)[["MAE", "RMSE", "R2", "MAE_low", "MAE_mid", "MAE_high"]]
                         .to_numpy(float), atol=1.5e-4) for k in cnn)

# ---------------------------------------------------------------- baseline (same as Step 3)
df = pd.read_csv(ROOT / "splits" / "splits.csv")
base, train_mean = {}, {}
for st in STRATS:
    col = f"{st}_split"
    m = df.loc[df[col] == "train", "L"].mean()
    train_mean[st] = m
    for sb in SETS:
        y = df.loc[df[col] == sb, "L"].to_numpy()
        base[(st, sb)] = metrics(y, np.full_like(y, m))

# set sizes must match the frozen splits
FROZEN = {("random", "val"): 6863, ("random", "test"): 6864,
          ("grouped", "val"): 6863, ("grouped", "test"): 6733}
sizes_ok = all(cnn[k]["n"] == FROZEN[k] for k in FROZEN)

# ---------------------------------------------------------------- history helpers
def hist_info(h, patience=3):
    ft = h[h.phase == "finetune"].reset_index(drop=True)
    best_i = int(h["val_mse"].idxmin())
    return {"n_head": int((h.phase == "head").sum()), "n_ft": len(ft),
            "best_phase": h.loc[best_i, "phase"], "best_ep": int(h.loc[best_i, "epoch"]),
            "best_mse": float(h.loc[best_i, "val_mse"]), "best_mae": float(h.loc[best_i, "val_mae"]),
            "min_mae": float(h["val_mae"].min()),
            "min_mae_ep": f"{h.loc[h['val_mae'].idxmin(), 'phase']} {int(h.loc[h['val_mae'].idxmin(), 'epoch'])}",
            "last_train": float(h["train_mse"].iloc[-1]), "best_i": best_i}


info = {st: hist_info(hist[st]) for st in STRATS}
train_min = {st: float(res_row(st, "test")["train_minutes"]) for st in STRATS}

# ---------------------------------------------------------------- figure 1: learning curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
for a, st in zip(axes, STRATS):
    h = hist[st]
    x = np.arange(1, len(h) + 1)
    lab = [f"H{e}" if ph == "head" else f"F{e}" for ph, e in zip(h.phase, h.epoch)]
    a.plot(x, h.train_mse, "o-", label="train MSE (augmented, train mode)")
    a.plot(x, h.val_mse, "s-", label="val MSE")
    a.axvline(info[st]["n_head"] + 0.5, ls="--", c="gray")
    bi = info[st]["best_i"]
    a.scatter([x[bi]], [h.val_mse.iloc[bi]], s=160, facecolors="none", edgecolors="red",
              linewidths=2, zorder=5, label="best checkpoint (lowest val MSE)")
    a.set_xticks(x); a.set_xticklabels(lab)
    a.set_title(f"resnet18_{st}  (H = head-only epoch, F = fine-tune epoch)", fontsize=10)
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
axes[0].set_ylabel("MSE (L as fraction)")
axes[0].legend(fontsize=8)
plt.tight_layout()
FIG_CURVES = FIG_DIR / "learning_curves.png"
plt.savefig(FIG_CURVES, dpi=120); plt.close()

# ---------------------------------------------------------------- figure 2: predicted vs true
fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
for a, (st, sb) in zip(axes, [(s, b) for s in STRATS for b in SETS]):
    d = preds[(st, sb)]
    a.scatter(d.L_true, d.L_pred, s=2, alpha=0.15)
    a.plot([0, 1], [0, 1], "r--", lw=1)
    a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02); a.set_aspect("equal")
    m = cnn[(st, sb)]
    a.set_title(f"{st} {sb}  (n={m['n']:,})\nMAE {m['MAE']:.4f}  R2 {m['R2']:.3f}", fontsize=10)
    a.set_xlabel("true L"); a.grid(alpha=0.3)
axes[0].set_ylabel("predicted L (clamped to [0, 1])")
plt.tight_layout()
FIG_SCAT = FIG_DIR / "pred_vs_true.png"
plt.savefig(FIG_SCAT, dpi=120); plt.close()

# ---------------------------------------------------------------- figure 3: per-range MAE vs baseline (test)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
xr = np.arange(3)
for a, st in zip(axes, STRATS):
    b = [base[(st, "test")][f"MAE_{k}"] for _, _, k, _ in BINS]
    c = [cnn[(st, "test")][f"MAE_{k}"] for _, _, k, _ in BINS]
    a.bar(xr - 0.2, b, 0.4, label="mean-predictor baseline", color="#B0B7C3")
    a.bar(xr + 0.2, c, 0.4, label="ResNet-18", color="#1F77B4")
    for i in range(3):
        a.text(xr[i] - 0.2, b[i] + 0.005, f"{b[i]:.3f}", ha="center", fontsize=8)
        a.text(xr[i] + 0.2, c[i] + 0.005, f"{c[i]:.3f}", ha="center", fontsize=8)
    a.set_xticks(xr); a.set_xticklabels(["low (L<0.15)", "mid (0.15-0.5)", "high (L>=0.5)"])
    a.set_title(f"{st} split, test set", fontsize=10); a.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("MAE (fraction)"); axes[0].legend(fontsize=8)
plt.tight_layout()
FIG_RANGE = FIG_DIR / "per_range_mae_test.png"
plt.savefig(FIG_RANGE, dpi=120); plt.close()

# ---------------------------------------------------------------- styles (same as Steps 1-3)
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=14,
                    spaceAfter=6, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10,
                    spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=5)
BL = ParagraphStyle("BL", parent=B, leftIndent=14, bulletIndent=4)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=8.5, leading=11,
                      backColor=colors.HexColor("#F3F4F6"), borderPadding=5,
                      leftIndent=4, spaceBefore=4, spaceAfter=8)
CELL = ParagraphStyle("CELL", parent=B, fontSize=9, leading=11, spaceAfter=0)
CELLS = ParagraphStyle("CELLS", parent=CELL, fontSize=8, leading=10)
C = "<font face='Courier'>%s</font>"
R2 = "R<super>2</super>"


def p(t):
    return Paragraph(t, B)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def table(rows, widths, header=True, small=False, shade_rows=None):
    st = CELLS if small else CELL
    d = [[Paragraph(str(c), st) for c in r] for r in rows]
    t = Table(d, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EAF2")))
    for r, c in (shade_rows or {}).items():
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor(c)))
    t.setStyle(TableStyle(style))
    return t


def f4(x):
    return f"{x:.4f}"


def ok(b):
    return "OK" if b else "FAILED"


def pct_lower(b, c):
    return 100 * (b - c) / b


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "IDSE project - Solar panel soiling power-loss regression - Step 4 documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


g, r = info["grouped"], info["random"]
gt, rt = cnn[("grouped", "test")], cnn[("random", "test")]
gv, rv = cnn[("grouped", "val")], cnn[("random", "val")]
ratio = gt["MAE"] / rt["MAE"]

# ---------------------------------------------------------------- content
s = []
s.append(Paragraph("Step 4: ResNet-18 Transfer-Learning Regression", ss["Title"]))
s.append(Paragraph(
    "Predicting soiling-induced power loss from solar panel images (DeepSolarEye)<br/>"
    "Introduction to Data Science for Engineers - IIITDM Kancheepuram", CAP))
s.append(Spacer(1, 8))
s.append(table([
    ["Member", "Roll no.", "Role"],
    ["Manieesh Manohar", "CS24B1005", "Project lead, data pipeline, splits, integration, report"],
    ["Mahalis Rayhan", "CS24B1095", "ResNet-18 model, training, evaluation"],
    ["Siddharth", "CS24B1013", "Dataset analysis, preprocessing, robustness tests"],
    ["Ram Niwas", "CS24I1001", "Grad-CAM, architecture explanation, slides"],
], [4 * cm, 2.6 * cm, 10.4 * cm]))
s.append(Spacer(1, 4))
s.append(Paragraph("Document date: 20 Nov 2026. Presentation: 21 Nov 2026. "
                   "Status: Step 4 <b>done</b>; Step 5 (detailed evaluation), Grad-CAM and the "
                   "robustness test are next.", CAP))

# 0 summary
s.append(Paragraph("Summary of results", H1))
s.append(table([
    ["Test set", "Baseline MAE", "ResNet-18 MAE", "MAE lower by", "Baseline " + R2, "ResNet-18 " + R2],
    ["<b>grouped (honest estimate)</b>", f4(base[("grouped", "test")]["MAE"]), f"<b>{f4(gt['MAE'])}</b>",
     f"{pct_lower(base[('grouped', 'test')]['MAE'], gt['MAE']):.0f}%",
     f"{base[('grouped', 'test')]['R2']:+.4f}", f"<b>{gt['R2']:.3f}</b>"],
    ["random (optimistic reference)", f4(base[("random", "test")]["MAE"]), f4(rt["MAE"]),
     f"{pct_lower(base[('random', 'test')]['MAE'], rt['MAE']):.0f}%",
     f"{base[('random', 'test')]['R2']:+.4f}", f"{rt['R2']:.3f}"],
], [4.6 * cm, 2.3 * cm, 2.5 * cm, 2.3 * cm, 2.6 * cm, 2.7 * cm]))
s.append(Spacer(1, 4))
s.extend(bullets([
    f"A pretrained ResNet-18 predicts the power loss L of an unseen <b>day</b> with a mean absolute "
    f"error of <b>{100 * gt['MAE']:.2f} percentage points</b> ({R2} {gt['R2']:.3f}), against "
    f"{100 * base[('grouped', 'test')]['MAE']:.2f} points for a model that ignores the image.",
    f"On the random split the same model looks much better ({100 * rt['MAE']:.2f} points, "
    f"{R2} {rt['R2']:.3f}). The grouped error is <b>{ratio:.2f}x</b> larger. This gap is the "
    "measured cost of temporal near-duplicate leakage found in Step 2; the grouped number is the "
    "headline result.",
]))

# 1 purpose
s.append(Paragraph("1. Purpose of Step 4 and where it fits", H1))
s.append(p(
    "Step 1 audited the data and fixed the target (L, power loss as a fraction 0-1). Step 2 froze a "
    "random split and a grouped-by-day split. Step 3 built the shared data pipeline (" +
    C % "data.py" + ") and the mean-predictor baseline. Step 4 trains the actual model:"))
s.extend(bullets([
    "an ImageNet-pretrained <b>ResNet-18</b> with a one-number regression output (predicted L);",
    "trained <b>twice with identical settings</b>, once on each split strategy, so that the only "
    "difference between the two results is how the data was split;",
    "evaluated on val and test with MAE, RMSE, " + R2 + " and MAE per L range, and compared with "
    "the baseline of the <b>same</b> split and set;",
    "with checkpoints and <b>per-image predictions</b> saved for Step 5 (error analysis), Grad-CAM "
    "and the robustness test.",
]))
s.append(p("Step 4 only reads the split files and the raw images; neither is changed."))

# 2 what & why
s.append(Paragraph("2. What we did and why", H1))
s.append(table([
    ["#", "Action", "Why"],
    ["4.1", "Wrote " + C % "train.py" + ": model, two-phase training, early stopping, evaluation, "
            "saving of history, curves, checkpoints, predictions and a results table.",
     "One script for both strategies; all settings are command-line arguments with fixed defaults, "
     "so a run is fully described by its command."],
    ["4.2", "Used a pretrained ResNet-18 and replaced its final layer by a single linear output.",
     "Transfer learning: ImageNet filters already detect edges, textures and bright blobs, which is "
     "what dust and deposits look like. Much cheaper and more data-efficient than training from "
     "scratch on 15 days of one camera."],
    ["4.3", "Phase A: froze the backbone and trained only the new head (2 epochs, lr 1e-3). "
            "Phase B: unfroze everything and fine-tuned with a 10x lower lr (1e-4).",
     "The new head starts random. Training it first stops its large early gradients from "
     "damaging the pretrained filters; the low lr in phase B then adapts the filters gently."],
    ["4.4", "Early stopping on validation MSE (patience 3), keeping the checkpoint with the "
            "lowest val MSE.",
     "Train error keeps falling while val error starts rising (memorising the training days). "
     "Stopping on val keeps the best generalising model without touching the test set."],
    ["4.5", "Ran a short debug run (" + C % "--limit 2000" + ") first.",
     "Checks the whole pipeline end to end (training, saving, reloading, evaluation, files) in "
     "about a minute before spending GPU time on the full runs."],
    ["4.6", "Trained the full model on the grouped split and on the random split.",
     "Quantifies how much a random split overstates performance on this time-series dataset."],
    ["4.7", "Evaluated the best checkpoint on val and test; saved per-image predictions.",
     "Reported metrics come from the selected model only; predictions allow later analysis "
     "(per day, worst cases, Grad-CAM example selection) without re-running the network."],
], [0.9 * cm, 7.6 * cm, 8.5 * cm]))

# 3 model
s.append(Paragraph("3. Model", H1))
s.append(table([
    ["Part", "Setting", "Why"],
    ["Backbone", "torchvision ResNet-18, weights " + C % "IMAGENET1K_V1",
     "Small (fits easily in 8 GB, ~34 s per epoch here), well understood, standard for transfer "
     "learning, and has a clear last conv block (" + C % "layer4" + ") for Grad-CAM."],
    ["Head", C % "fc = Linear(512, 1)" + " replacing the 1000-class layer",
     "Regression of one continuous value (L), not classification. The 512-d global-average-pooled "
     "features are mapped to one number."],
    ["Output", "raw linear value during training; clamped to [0, 1] only at evaluation",
     "L is a fraction, so predictions outside [0, 1] are physically impossible and clamping can "
     "only reduce error. No sigmoid is used in training, so gradients do not vanish near 0 and 1 "
     "(many labels are near 0)."],
    ["Size", f"{n_params / 1e6:.2f} M trainable parameters (results.csv: "
             f"{res_row('grouped', 'test')['params_M']} M)", "All layers are trained in phase B."],
    ["Input", "3 x 224 x 224, from " + C % "data.py" + " (fixed Step 3 preprocessing)",
     "Resize 192 to 224, ImageNet normalisation, train-only brightness/contrast jitter 0.15. "
     "Irradiance I is <b>not</b> an input."],
], [2.2 * cm, 5.6 * cm, 9.2 * cm], small=True))

# 4 training procedure
s.append(Paragraph("4. Training procedure and settings", H1))
s.append(table([
    ["Setting", "Value", "Why"],
    ["Loss", "MSE on L (fraction)", "Standard regression loss; punishes large errors more, which "
     "matters for badly mis-judged heavy soiling."],
    ["Optimiser", "AdamW", "Adaptive steps work well for fine-tuning; decoupled weight decay."],
    ["Phase A (head)", "2 epochs, lr 1e-3, backbone frozen (requires_grad = False), AdamW default "
     "weight decay 0.01 on the 513 head parameters", "Fit the new head quickly on fixed features."],
    ["Phase B (fine-tune)", "up to 12 epochs, lr 1e-4, weight decay 1e-4, all layers",
     "Lower lr preserves the pretrained features; weight decay is mild regularisation."],
    ["Early stopping", "patience 3 on val MSE (phase B only)", "Stop after 3 fine-tune epochs "
     "without a new best val MSE."],
    ["Checkpoint", "lowest val MSE over <b>both</b> phases", "Selection only uses val; test is "
     "evaluated once, at the end."],
    ["Batch size", "64", "Fits comfortably in 8 GB with bf16; stable gradient estimates."],
    ["Precision", "bf16 autocast on GPU; loss computed in float32",
     "About 2x faster and less memory on the RTX 5060. bf16 has the same exponent range as float32, "
     "so no gradient scaler is needed."],
    ["Data loading", "4 worker processes, pinned memory, persistent workers; train shuffled",
     "See 4.2 below."],
    ["Learning-rate schedule", "none (constant lr in each phase)", "Kept simple given the short "
     "runs; early stopping controls the length."],
    ["Seed", "42 (torch and numpy)", "Same initialisation and shuffling order for both runs."],
    ["Device", "NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.11.0+cu128", "Environment from Step 3."],
], [3 * cm, 6.2 * cm, 7.8 * cm], small=True))
s.append(Paragraph("4.1 Commands", H2))
s.append(Preformatted(
    ".\\.venv\\Scripts\\Activate.ps1\n"
    "python train.py --strategy grouped --limit 2000 --epochs 2 --head-epochs 1   # debug run\n"
    "python train.py --strategy grouped                                          # full run\n"
    "python train.py --strategy random                                           # full run",
    CODE))
s.append(p("Arguments stored in the grouped checkpoint (identical for random except " +
           C % "strategy" + "): " + C % ", ".join(f"{k}={v}" for k, v in ck_args["grouped"].items()
                                                    if k != "strategy") + "."))
s.append(Paragraph("4.2 Data loading speed (open point from Step 3)", H2))
s.append(p(
    "Step 3 measured ~301 images/s with 0 workers (about 1.8 min of loading alone per grouped-train "
    "epoch), while the 4-worker timing there was a start-up artifact. In the full runs with 4 "
    f"persistent workers, a whole epoch (loading, training and validation) took about 34 s, and the "
    f"complete grouped run took {train_min['grouped']} min for {g['n_head'] + g['n_ft']} epochs. "
    "4 workers are therefore clearly faster and were kept. A separate full training run with 0 "
    "workers was not done, so no exact speed-up factor is claimed."))
s.append(Paragraph("4.3 Exact epoch loop", H2))
s.extend(bullets([
    "Each epoch: one pass over the shuffled training set (with jitter) in train mode; then the "
    "whole val set is predicted in eval mode (no jitter, predictions clamped) and val MSE and MAE "
    "are computed.",
    "If val MSE is the lowest so far, the weights and the run arguments are saved to " +
    C % "outputs/checkpoints/&lt;run&gt;_best.pt" + ".",
    "After training: history CSV and curve plot are saved, the best checkpoint is reloaded, "
    "val and test are predicted once, per-image predictions are written, and one row per set is "
    "appended to " + C % "outputs/results.csv" + ".",
]))

# 5 debug run
s.append(Paragraph("5. Debug run (4.5)", H1))
dv = res[res.run == "resnet18_grouped_lim2000"]
s.append(p(
    C % "--limit 2000" + " takes a random sample of 2,000 images from each of train, val and test "
    "(grouped split, sampling seed 0), with 1 head epoch and 2 fine-tune epochs."))
rows = [["Phase", "Epoch", "Train MSE", "Val MSE", "Val MAE"]]
for _, h in hist_dbg.iterrows():
    rows.append([h.phase, int(h.epoch), f4(h.train_mse), f4(h.val_mse), f4(h.val_mae)])
s.append(table(rows, [3 * cm, 2 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm], small=True))
s.append(Spacer(1, 4))
dvv = dv[dv.set == "val"].iloc[0]; dtt = dv[dv.set == "test"].iloc[0]
s.append(p(
    f"Result on the 2,000-image subsets: val MAE {f4(dvv.MAE)} ({R2} {dvv.R2:.3f}), test MAE "
    f"{f4(dtt.MAE)} ({R2} {dtt.R2:.3f}); training time {dvv.train_minutes} min. Purpose was only "
    "to prove that the pipeline runs end to end and that the loss falls; these numbers are "
    "<b>not</b> results. Its files carry the suffix " + C % "_lim2000" + " so they cannot be "
    "confused with the full runs."))

# 6 training histories
s.append(Paragraph("6. Training histories", H1))
for st in STRATS:
    h, i = hist[st], info[st]
    s.append(Paragraph(f"6.{1 if st == 'grouped' else 2} {st.capitalize()} split "
                       f"(" + C % f"resnet18_{st}" + ")", H2))
    rows = [["Phase", "Epoch", "Train MSE", "Val MSE", "Val MAE", "Note"]]
    shade = {}
    for j, row in h.iterrows():
        note = ""
        if j == i["best_i"]:
            note = "<b>best (checkpoint kept)</b>"; shade[j + 1] = "#E8F4EA"
        elif j == len(h) - 1:
            note = "3rd epoch without improvement: early stop"
        rows.append([row.phase, int(row.epoch), f4(row.train_mse), f4(row.val_mse),
                     f4(row.val_mae), note])
    s.append(table(rows, [2.2 * cm, 1.5 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 6.1 * cm],
                   small=True, shade_rows=shade))
    s.append(Spacer(1, 3))
    s.append(p(f"{i['n_head']} head + {i['n_ft']} fine-tune epochs, early stopped; best = "
               f"{i['best_phase']} epoch {i['best_ep']} (val MSE {i['best_mse']:.4f}, val MAE "
               f"{i['best_mae']:.4f}). Total training time {train_min[st]} min."))
s.append(KeepTogether([
    Image(str(FIG_CURVES), width=17 * cm, height=17 * cm * 4.3 / 13),
    Paragraph("Figure 1. Train and val MSE per epoch for both runs (from the history CSVs). Dashed "
              "line: switch from head-only to full fine-tuning. Red circle: kept checkpoint.", CAP)]))
s.append(Paragraph("6.3 What the curves show", H2))
s.extend(bullets([
    "<b>The head alone already learns a lot.</b> After 2 head-only epochs, val MAE is about 0.09 "
    "on both splits (baseline about 0.25). Frozen ImageNet features already separate clean from "
    "soiled panels.",
    "<b>Fine-tuning helps further</b>, most clearly on the random split (val MSE 0.0163 to 0.0085 "
    "after one fine-tune epoch).",
    f"<b>Overfitting on the grouped split.</b> Train MSE falls steadily to {g['last_train']:.4f}, "
    f"but grouped val MSE is noisy (0.0138, 0.0142, 0.0172, 0.0120, ...) and rises to "
    f"{hist['grouped'].val_mse.iloc[-1]:.4f} by the last epoch. The model starts to memorise the 10 "
    "training days; early stopping kept fine-tune epoch 4.",
    f"<b>On the random split val keeps tracking train</b> (best val MSE {r['best_mse']:.4f}, much "
    f"lower than grouped {g['best_mse']:.4f}). Random val images have a training frame about 5 s "
    "away (Step 2), so memorising training frames still helps on val. This is the leakage, seen in "
    "the curves.",
    "The train MSE is measured with augmentation and in train mode during the epoch, so it is "
    "not directly comparable to val MSE; the trend is what matters.",
]))

# 7 results
s.append(Paragraph("7. Results of the selected checkpoints", H1))
s.append(p("All metrics are on L as a fraction (multiply MAE/RMSE by 100 for percentage points). "
           "Baseline = predict the training-set mean L of the same strategy (Step 3)."))
rows = [["Run", "Set", "n", "Model", "MAE", "MAE (pts)", "RMSE", R2]]
shade = {}
for st in STRATS:
    for sb in SETS:
        for name, mm in (("baseline", base[(st, sb)]), ("ResNet-18", cnn[(st, sb)])):
            rows.append([st, sb, f"{mm['n']:,}", name, f4(mm["MAE"]), f"{100 * mm['MAE']:.2f}",
                         f4(mm["RMSE"]), f"{mm['R2']:+.4f}"])
            if name == "ResNet-18":
                shade[len(rows) - 1] = "#E8F4EA"
s.append(table(rows, [1.9 * cm, 1.2 * cm, 1.5 * cm, 2.3 * cm, 1.9 * cm, 2 * cm, 1.9 * cm, 2 * cm],
               small=True, shade_rows=shade))
s.append(Paragraph(f"Metrics recomputed from outputs/preds/*.csv agree with outputs/results.csv: "
                   f"{ok(csv_ok)}. Set sizes match the frozen splits: {ok(sizes_ok)}.", CAP))

s.append(Paragraph("7.1 MAE and bias per L range", H2))
s.append(p("L is strongly right-skewed, so the error is also reported per range. Bias = mean(predicted "
           "- true): negative means the model under-predicts the loss."))
rows = [["Run", "Set", "L range", "n", "Baseline MAE", "ResNet-18 MAE", "ResNet-18 bias"]]
shade = {}
for st in STRATS:
    for sb in SETS:
        for lo, hi, k, name in BINS:
            rows.append([st, sb, name, f"{cnn[(st, sb)][f'n_{k}']:,}", f4(base[(st, sb)][f"MAE_{k}"]),
                         f4(cnn[(st, sb)][f"MAE_{k}"]), f"{cnn[(st, sb)][f'bias_{k}']:+.4f}"])
            if cnn[(st, sb)][f"MAE_{k}"] > base[(st, sb)][f"MAE_{k}"]:
                shade[len(rows) - 1] = "#FCE8E6"
s.append(table(rows, [1.9 * cm, 1.2 * cm, 3.2 * cm, 1.6 * cm, 2.8 * cm, 3 * cm, 3.1 * cm],
               small=True, shade_rows=shade))
s.append(Paragraph("Red rows: the ResNet-18 MAE is higher than the baseline MAE in that range.", CAP))
s.append(KeepTogether([
    Image(str(FIG_RANGE), width=16 * cm, height=16 * cm * 3.9 / 12),
    Paragraph("Figure 2. Test MAE per L range, baseline vs ResNet-18.", CAP)]))
s.append(KeepTogether([
    Image(str(FIG_SCAT), width=17 * cm, height=17 * cm * 4.3 / 16),
    Paragraph("Figure 3. Predicted vs true L for every val/test image (red dashed line = perfect "
              "prediction). Made from the saved prediction files.", CAP)]))

# 8 interpretation
s.append(Paragraph("8. Interpretation", H1))
s.append(Paragraph("8.1 The model learned real image information", H2))
s.extend(bullets([
    f"Grouped test: MAE {f4(base[('grouped', 'test')]['MAE'])} to {f4(gt['MAE'])} "
    f"({pct_lower(base[('grouped', 'test')]['MAE'], gt['MAE']):.0f}% lower), {R2} "
    f"{base[('grouped', 'test')]['R2']:+.4f} to {gt['R2']:.3f}. These are 2 days the model never saw.",
    f"The biggest gains are where the baseline fails: clean panels (low range, grouped test "
    f"{f4(base[('grouped', 'test')]['MAE_low'])} to {f4(gt['MAE_low'])}) and heavy soiling (high "
    f"range {f4(base[('grouped', 'test')]['MAE_high'])} to {f4(gt['MAE_high'])}). The model "
    "therefore distinguishes clean from heavily soiled panels rather than guessing near the mean.",
]))
s.append(Paragraph("8.2 Random split vs grouped split (main finding)", H2))
s.append(table([
    ["", "Random split", "Grouped split"],
    ["Test MAE", f4(rt["MAE"]), f4(gt["MAE"])],
    ["Test RMSE", f4(rt["RMSE"]), f4(gt["RMSE"])],
    ["Test " + R2, f"{rt['R2']:.3f}", f"{gt['R2']:.3f}"],
    ["Median time from a test image to its nearest train image (Step 2)", "5 s", "~18 h (66,214 s)"],
    ["Test images within 10 s of a train image (Step 2)", "92.5%", "0.0%"],
], [8 * cm, 4.5 * cm, 4.5 * cm], small=True))
s.append(Spacer(1, 4))
s.extend(bullets([
    f"Same model, same settings, same seed: only the split differs. The grouped test MAE is "
    f"<b>{ratio:.2f}x</b> the random test MAE and {R2} drops from {rt['R2']:.3f} to {gt['R2']:.3f}.",
    "With a random split, almost every test frame has a nearly identical training frame taken "
    "seconds earlier (same dust, same light). The model can succeed by recognising the scene "
    "rather than judging the soiling, so the random score measures <b>interpolation within known "
    "days</b>, not prediction for a new day.",
    "The grouped split is the realistic deployment setting (a new day's image) and is the "
    "<b>headline result</b>. The random result is shown only as an optimistic reference.",
    f"On the random split val and test agree almost exactly (MAE {f4(rv['MAE'])} vs "
    f"{f4(rt['MAE'])}), because both are drawn from the same days as train. On the grouped split "
    f"they differ (val {f4(gv['MAE'])}, {R2} {gv['R2']:.3f}; test {f4(gt['MAE'])}, {R2} "
    f"{gt['R2']:.3f}), because val and test are different days with different conditions.",
]))
s.append(Paragraph("8.3 Weak points found", H2))
s.extend(bullets([
    f"<b>Mid range (0.15-0.5) on the grouped split:</b> ResNet-18 is <b>worse than the baseline</b> "
    f"on both grouped val (MAE {f4(gv['MAE_mid'])} vs {f4(base[('grouped', 'val')]['MAE_mid'])}) "
    f"and grouped test ({f4(gt['MAE_mid'])} vs {f4(base[('grouped', 'test')]['MAE_mid'])}). The "
    "constant (~0.27) lies inside this range, so the baseline is naturally hard to beat there, but "
    "it also shows that moderate soiling levels are the hardest to judge on a new day. On the random "
    f"split the model does beat the baseline in this range ({f4(rt['MAE_mid'])} vs "
    f"{f4(base[('random', 'test')]['MAE_mid'])} on test), which again points to leakage helping "
    "the random split.",
    f"<b>Pull towards the middle:</b> on every set the bias is positive for clean panels and "
    f"negative for mid and high soiling (grouped test {gt['bias_low']:+.3f} / {gt['bias_mid']:+.3f} / "
    f"{gt['bias_high']:+.3f}), i.e. the model over-predicts low losses and under-predicts high ones. "
    "This is typical for MSE regression on noisy labels.",
    f"<b>Clean panels on the grouped test days:</b> low-range MAE {f4(gt['MAE_low'])} vs "
    f"{f4(gv['MAE_low'])} on grouped val, with bias {gt['bias_low']:+.4f} (the model over-predicts "
    "loss on clean panels on these days). Per-day analysis in Step 5 will show whether one test day "
    "(21 or 29 Jun) causes this.",
    f"<b>Heavy soiling on grouped val:</b> high-range MAE {f4(gv['MAE_high'])}, bias "
    f"{gv['bias_high']:+.4f}; bias equals minus the MAE, so essentially every heavily soiled val "
    "image is under-predicted. The grouped test set has almost no L above 0.8 "
    "(Step 2), so very heavy soiling is judged mainly on val, where it is the weakest range.",
    "<b>Structure in Figure 3 (grouped):</b> predictions form tight clusters (consecutive frames of "
    "the same day look alike and get similar predictions), and grouped val clusters at true L ~0.55 "
    "and ~0.8 sit clearly below the diagonal. On grouped test a horizontal band of images with true "
    "L between about 0.2 and 1.0 is predicted near 0.1, i.e. they look clean to the model. These "
    "are the failure cases to inspect in Step 5 (worst-20 list) and with Grad-CAM.",
    "<b>Label noise</b> (Step 3: a few rust spots with L 0.891 vs large white patches with L "
    "0.731) puts a ceiling on achievable accuracy; the vertical spread in Figure 3 is partly this.",
    f"Share of predictions clamped to exactly 0 on grouped test: {100 * gt['clamped0']:.1f}% "
    f"(to 1: {100 * gt['clamped1']:.1f}%); the clamp only affects the evaluation, not training.",
]))

# 9 outputs
s.append(Paragraph("9. Outputs produced", H1))
s.append(table([
    ["File", "Content"],
    ["train.py", "Model, two-phase training, early stopping, evaluation and saving (all settings as "
                 "arguments)."],
    ["outputs/checkpoints/resnet18_{grouped,random}_best.pt",
     "Best weights (lowest val MSE) plus the run arguments. The grouped checkpoint is the one used "
     "for Grad-CAM and robustness."],
    ["outputs/preds/resnet18_{grouped,random}_{val,test}.csv",
     "One row per image: filename, L_true, L_pred (clamped). Input to Step 5."],
    ["outputs/resnet18_{grouped,random}_history.csv", "phase, epoch, train_mse, val_mse, val_mae."],
    ["outputs/plots/resnet18_{grouped,random}_curves.png", "Train/val MSE curves per run."],
    ["outputs/results.csv", "One row per run and set: n, MAE, RMSE, R2, MAE_low/mid/high, "
                            "train_minutes, params_M (debug run rows included, tagged _lim2000)."],
    ["outputs/*_lim2000*", "Debug-run files (not results)."],
    ["docs/step4_figs/", "Figures 1-3 of this document (made by docs/build_step4_pdf.py)."],
], [7.2 * cm, 9.8 * cm], small=True))

# 10 decisions & caveats
s.append(Paragraph("10. Decisions and caveats", H1))
s.extend(bullets([
    "<b>Headline = grouped test</b> (MAE " + f4(gt["MAE"]) + ", " + R2 + f" {gt['R2']:.3f}). "
    "The random-split result is reported only as an optimistic reference showing the leakage effect.",
    "<b>Test was not used for any decision.</b> Checkpoints were selected on val MSE; test was "
    "evaluated once at the end. Grouped val numbers are slightly optimistic because they were used "
    "for selection; only the grouped test result is truly held out. No further tuning on test.",
    "<b>Small evaluation sets in days:</b> grouped test is only 2 days (21 and 29 Jun) and val 3 days, "
    "so the estimates carry real uncertainty and depend on those days' weather and soiling.",
    "<b>Single seed (42), single run per split.</b> Run-to-run variation was not measured. Also, "
    "cuDNN was not set to deterministic mode and bf16 is used, so a re-run will give close but not "
    "bit-identical numbers.",
    "<b>Selection on MSE, not MAE:</b> the epoch with the lowest val MSE is not always the one with "
    f"the lowest val MAE (grouped: lowest val MAE {g['min_mae']:.4f} at {g['min_mae_ep']}; random: "
    f"{r['min_mae']:.4f} at {r['min_mae_ep']}). MSE was fixed as the criterion before training, "
    "matching the loss, and was not changed after seeing the results.",
    "<b>Noisy grouped validation curve:</b> val MSE jumps between epochs (e.g. 0.0142 to 0.0172 to "
    "0.0120), so the chosen epoch is partly luck; with patience 3 the run still covered the "
    "improvement at fine-tune epoch 4.",
    "<b>Phase A detail:</b> the backbone weights are frozen, but the model is in train mode, so the "
    "BatchNorm running statistics are still updated to this dataset during the head epochs. This is "
    "standard behaviour and harmless here, but it means phase A is not a pure 'fixed feature "
    "extractor'.",
    "<b>Constant learning rate</b> in each phase (no scheduler); not tuned, to stay within the time "
    "budget. No hyperparameter search was done.",
    "<b>One site, one camera, 15 days</b>: results may not transfer to other panels or cameras.",
    "<b>results.csv is appended to</b> by every run; re-running a training adds new rows instead of "
    "replacing old ones (this document reads the last row per run).",
]))

# 11 next
s.append(Paragraph("11. Next steps", H1))
s.extend(bullets([
    "<b>Step 5 - " + C % "evaluate.py" + "</b>: comparison table, error and bias per L range, "
    "per-day error (explain the grouped-test clean-panel error), scatter plots, worst-20 lists of "
    "failure cases.",
    "<b>Grad-CAM</b> on " + C % "resnet18_grouped_best.pt" + " (scalar output, " + C % "layer4" +
    "): clean / moderate / heavy and well / badly predicted examples; check that the model looks at "
    "the panel and not the background (e.g. the image with a person next to the panel).",
    "<b>Robustness test</b> on the grouped test set: brightness, Gaussian noise and JPEG compression "
    "perturbations, compared with the clean test MAE.",
    "Optional: EfficientNet-B0 only if time allows. Then report, slides and viva preparation.",
]))
s.append(KeepTogether([Paragraph("11.1 Timeline", H2), table([
    ["Date", "Work"],
    ["20 Nov 2026", "Step 5 evaluation, Grad-CAM, robustness test; code freeze and report; viva "
                    "preparation in the evening"],
    ["21 Nov 2026", "Presentation"],
], [4 * cm, 13 * cm])]))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Step 4: ResNet-18 Transfer-Learning Regression",
                        author="IDSE Solar Soiling Team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("results.csv matches preds:", csv_ok, "| sizes ok:", sizes_ok, "| params:", n_params)
for k, v in cnn.items():
    print(k, {m: round(float(v[m]), 4) for m in ("MAE", "RMSE", "R2", "MAE_low", "MAE_mid",
                                                  "MAE_high", "bias_low", "bias_mid", "bias_high",
                                                  "clamped0", "clamped1")})
print("Wrote", PDF)
