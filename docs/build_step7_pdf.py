"""Build docs/Step7_Robustness_and_Occlusion.pdf (Step 7 documentation).

Usage (from the project folder, venv active):
    python docs/build_step7_pdf.py

No model is run here. All numbers come from the files written by robustness.py
(outputs/eval/robustness_grouped_test*.csv). The clean row is cross-checked against the Step 4
prediction file. Three illustrative figures are made here on the CPU:
  * the occlusion boxes drawn on one test image,
  * one test image under every perturbation (using the perturbation functions of robustness.py),
  * MAE versus perturbation strength (plotted from the CSV).
Raw images are only read.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import robustness as rb  # noqa: E402  (perturbation functions, read only)
from data import DATA_ROOT  # noqa: E402

EV = ROOT / "outputs" / "eval"
FIG_DIR = ROOT / "docs" / "step7_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF = ROOT / "docs" / "Step7_Robustness_and_Occlusion.pdf"

# ---------------------------------------------------------------- load
res = pd.read_csv(EV / "robustness_grouped_test.csv")
ctl = pd.read_csv(EV / "robustness_grouped_test_controls.csv")
dbg = pd.read_csv(EV / "robustness_grouped_test_lim500.csv")
R = res.set_index("condition")
CT = ctl.set_index("condition")
DB = dbg.set_index("condition")
sp = pd.read_csv(ROOT / "splits" / "splits.csv", parse_dates=["timestamp"])
gtest = sp[sp.grouped_split == "test"]
pr = pd.read_csv(ROOT / "outputs" / "preds" / "resnet18_grouped_test.csv")
y, p = pr.L_true.to_numpy(float), pr.L_pred.to_numpy(float)
e = y - p
saved = {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
         "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), "bias": (p - y).mean()}
clean = R.loc["clean (no change)"]
clean_ok = (abs(clean.MAE - saved["MAE"]) < 1e-4 and abs(clean.RMSE - saved["RMSE"]) < 1e-4
            and abs(clean.R2 - saved["R2"]) < 1e-4 and abs(clean["bias(pred-true)"] - saved["bias"]) < 1e-4)
ctl_clean_ok = np.allclose(CT.loc["clean (no change)"].to_numpy(float), clean.to_numpy(float))
n_test = len(gtest)
BASE_MAE, BASE_R2 = 0.2100, -0.0025  # mean-predictor baseline, grouped test (Step 3)
SD_TEST = gtest.L.std()  # spread of labels on grouped test


def rel(c, df=R):
    return 100 * (df.loc[c, "MAE"] / clean.MAE - 1)


# ---------------------------------------------------------------- figure A: occlusion boxes
ex = gtest[(gtest.L.between(0.6, 0.7)) & (gtest.day == "2017-06-29")].sample(1, random_state=0).iloc[0]
img224 = T.Resize((224, 224))(PILImage.open(DATA_ROOT / ex.filename).convert("RGB"))
boxes = [  # (label, y0, y1, x0, x1, colour)
    ("bottom 20% (stand)", 179, 224, 0, 224, "#D62728"),
    ("top-right corner", 0, 56, 168, 224, "#FF7F0E"),
    ("panel centre", 84, 140, 84, 140, "#9467BD"),
    ("control: top-left", 0, 56, 0, 56, "#2CA02C"),
    ("control: bottom-left", 168, 224, 0, 56, "#17BECF"),
    ("control: top-centre", 0, 56, 84, 140, "#1F77B4"),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
axes[0].imshow(img224)
for lab, y0, y1, x0, x1, c in boxes:
    axes[0].add_patch(mpatches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                                         ec=c, lw=2.2, ls="--" if lab.startswith("control") else "-"))
axes[0].legend(handles=[mpatches.Patch(ec=c, fc="none", lw=2, label=lab) for lab, *_, c in boxes],
               fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
axes[0].set_title("All occlusion regions (224x224 input)", fontsize=10)
axes[0].axis("off")
x0 = T.PILToTensor()(img224).unsqueeze(0).float() / 255
both = rb.occlude(bottom=True, topright=True)(x0, None)
both = rb.box(84, 140, 84, 140)(both, None)
axes[1].imshow(both[0].permute(1, 2, 0).numpy())
axes[1].set_title("Grey (0.5) fill: stand + top-right + panel centre", fontsize=10)
axes[1].axis("off")
plt.tight_layout()
FIG_OCC = FIG_DIR / "occlusion_regions.png"
plt.savefig(FIG_OCC, dpi=120); plt.close()

# ---------------------------------------------------------------- figure B: perturbation gallery
g = torch.Generator(); g.manual_seed(42)
gallery = [c for c in res.condition if not c.startswith("occlude")]
fig, axes = plt.subplots(3, 5, figsize=(13, 8.4))
for a, (name, fn) in zip(axes.ravel(), [(n, f) for n, f in rb.CONDITIONS if n in gallery]):
    g.manual_seed(42)
    out = fn(x0.clone(), g)
    a.imshow(out[0].permute(1, 2, 0).clamp(0, 1).numpy())
    a.set_title(f"{name}\nMAE on test {R.loc[name, 'MAE']:.4f}", fontsize=8.5)
    a.axis("off")
plt.tight_layout()
FIG_GAL = FIG_DIR / "perturbation_gallery.png"
plt.savefig(FIG_GAL, dpi=110); plt.close()

# ---------------------------------------------------------------- figure C: severity curves
fam = {
    "brightness factor": [(0.6, "brightness x0.6"), (0.8, "brightness x0.8"), (1.0, "clean (no change)"),
                          (1.2, "brightness x1.2"), (1.4, "brightness x1.4")],
    "contrast factor": [(0.6, "contrast x0.6"), (1.0, "clean (no change)"), (1.4, "contrast x1.4")],
    "noise sigma": [(0, "clean (no change)"), (0.02, "noise sigma 0.02"), (0.05, "noise sigma 0.05"),
                    (0.10, "noise sigma 0.10")],
    "blur sigma (px)": [(0, "clean (no change)"), (1, "blur sigma 1"), (2, "blur sigma 2")],
    "JPEG quality": [(100, "clean (no change)"), (50, "JPEG q50"), (20, "JPEG q20"), (10, "JPEG q10")],
}
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for j, (xl, pts) in enumerate(fam.items()):
    xs = [a for a, _ in pts]
    for i, (col, yl) in enumerate((("MAE", "MAE"), ("bias(pred-true)", "bias (pred - true)"))):
        a = axes[i, j]
        a.plot(xs, [R.loc[c, col] for _, c in pts], "o-", c="#1F77B4")
        if col == "MAE":
            a.axhline(clean.MAE, ls="--", c="gray", lw=1, label="clean")
            a.axhline(BASE_MAE, ls=":", c="#D62728", lw=1, label="mean baseline")
            a.set_ylim(0, 0.26)
        else:
            a.axhline(0, c="k", lw=0.6)
            a.set_ylim(-0.13, 0.25)
        if xl == "JPEG quality":
            a.invert_xaxis()
        a.set_xlabel(xl + (" (100 = original)" if xl == "JPEG quality" else ""), fontsize=8.5)
        a.set_ylabel(yl, fontsize=8.5)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
axes[0, 0].legend(fontsize=7)
plt.suptitle("Grouped test set: MAE (top) and bias (bottom) versus perturbation strength", fontsize=11)
plt.tight_layout()
FIG_SEV = FIG_DIR / "severity_curves.png"
plt.savefig(FIG_SEV, dpi=110); plt.close()

# ---------------------------------------------------------------- styles (same as Steps 1-6)
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


def para(t):
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


def ok(b):
    return "OK" if b else "FAILED"


def fig(path, w_cm, caption):
    with PILImage.open(path) as im:
        w, h = im.size
    return KeepTogether([Image(str(path), width=w_cm * cm, height=w_cm * cm * h / w),
                         Paragraph(caption, CAP)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "IDSE project - Solar panel soiling power-loss regression - Step 7 documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def f4(x):
    return f"{x:.4f}"


def res_table(df, names=None):
    rows = [["Condition", "MAE", "RMSE", R2, "Bias (pred-true)", "Mean |pred. shift|",
             "MAE change vs clean"]]
    shade = {}
    for c in (names or df.index):
        r = df.loc[c]
        d = 100 * (r.MAE / clean.MAE - 1)
        rows.append([c, f4(r.MAE), f4(r.RMSE), f"{r.R2:.4f}", f"{r['bias(pred-true)']:+.4f}",
                     f4(r["mean|pred shift|"]), "-" if c.startswith("clean") else f"{d:+.1f}%"])
        if d > 25:
            shade[len(rows) - 1] = "#FCE8E6"
        elif d < 0 and not c.startswith("clean"):
            shade[len(rows) - 1] = "#E8F4EA"
    return rows, shade


W7 = [5.1 * cm, 1.6 * cm, 1.6 * cm, 1.7 * cm, 2.3 * cm, 2.3 * cm, 2.4 * cm]

# ---------------------------------------------------------------- content
s = []
s.append(Paragraph("Step 7: Robustness and Occlusion Tests", ss["Title"]))
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
s.append(Paragraph("Document date: 20 Nov 2026. Presentation: 21 Nov 2026. Status: Step 7 <b>done</b>. "
                   "All planned experiments are complete; next: report and slides.", CAP))

# summary
s.append(Paragraph("Summary of results", H1))
s.extend(bullets([
    f"<b>Setup:</b> the trained grouped-split ResNet-18 (no retraining, no tuning) was evaluated on "
    f"the full grouped test set (n = {n_test:,}, 21 and 29 Jun) after applying 14 photometric, noise, "
    f"blur and compression perturbations and 6 occlusions (3 hypothesis boxes + 3 controls). Clean "
    f"reference: MAE {f4(clean.MAE)}, RMSE {f4(clean.RMSE)}, {R2} {clean.R2:.4f}.",
    f"<b>Robust</b> (MAE change at most about 11%) to brightness x0.6-1.4, contrast x0.6-1.4, light "
    f"noise (sigma 0.02), light blur (sigma 1) and JPEG quality 50.",
    f"<b>Sensitive</b> to strong noise (sigma 0.10: MAE {f4(R.loc['noise sigma 0.10', 'MAE'])}, "
    f"{rel('noise sigma 0.10') / 100 + 1:.1f}x, {R2} {R.loc['noise sigma 0.10', 'R2']:.2f}, worse than "
    f"the mean baseline MAE {BASE_MAE:.4f}), blur sigma 2 ({rel('blur sigma 2'):+.0f}%), JPEG q20 "
    f"({rel('JPEG q20'):+.0f}%) and q10 ({rel('JPEG q10'):+.0f}%).",
    "<b>Direction of the errors:</b> noise and JPEG artefacts <b>raise</b> the predicted loss "
    "(speckle looks like dirt); blur <b>lowers</b> it (smooths away deposits); brighter images raise "
    "it slightly.",
    f"<b>Occlusion (testing Step 6 Grad-CAM):</b> hiding the stand does not hurt (MAE "
    f"{f4(R.loc['occlude bottom 20%', 'MAE'])}); grey boxes in background corners have no effect; "
    f"hiding the panel centre (6% of the image) doubles the error (MAE "
    f"{f4(CT.loc['occlude panel centre 56x56', 'MAE'])}, {R2} "
    f"{CT.loc['occlude panel centre 56x56', 'R2']:.2f}); hiding the top-right corner raises MAE by "
    f"{rel('occlude top-right corner'):.0f}%. <b>Predictions depend mainly on the panel area.</b>",
]))
s.append(Paragraph(f"Check: the clean row reproduces the Step 4 grouped test metrics from the saved "
                   f"predictions: {ok(clean_ok)}; the controls run has the identical clean row: "
                   f"{ok(ctl_clean_ok)}.", CAP))

# 1 purpose
s.append(Paragraph("1. Purpose of Step 7 and where it fits", H1))
s.append(para("Step 7 answers two separate questions with the same tool (change the test images, keep "
              "the model fixed, measure the error):"))
s.extend(bullets([
    "<b>Robustness:</b> how much do the predictions change when imaging conditions change? A deployed "
    "monitoring camera sees different exposure, sensor noise, focus/haze and compression. The "
    "DeepSolarEye frames are small (192x192) and already JPEG-compressed (about 17 KB each) from one "
    "fixed camera, so the model has only ever seen one imaging setup. The survey lists robustness and "
    "domain shift as a main concern for image-based soiling estimation.",
    "<b>Background reliance (occlusion):</b> Step 6 Grad-CAM showed heat on the stand below the "
    "panel and on the top-right corner, even for clean panels. Grad-CAM is not causal, so the "
    "hypotheses 'the model depends on the stand' and 'the model depends on the top-right corner' are "
    "tested directly by hiding those regions. Control boxes check that a grey box by itself does not "
    "cause errors.",
]))
s.append(para(
    "The grouped model and grouped test set are used because they give the honest, held-out headline "
    "result. Nothing learned in this step is fed back into the model: these are diagnostics only."))

# 2 what & why
s.append(Paragraph("2. What we did and why", H1))
s.append(table([
    ["#", "Action", "Why"],
    ["7.1", "Wrote " + C % "robustness.py" + ": loads the grouped test images once (resized to "
     "224x224, uint8, in memory), applies each perturbation on the GPU, normalises, predicts, "
     "computes metrics.",
     "Loading 6,733 images once and reusing them makes 18 full-test evaluations fast (a few "
     "minutes); every condition sees exactly the same images."],
    ["7.2", "Photometric: brightness x0.6/0.8/1.2/1.4, contrast x0.6/1.4.",
     "Exposure and weather changes (cloud, time of day, camera auto-exposure)."],
    ["7.3", "Gaussian noise sigma 0.02/0.05/0.10.", "Sensor noise in low light or a cheaper camera."],
    ["7.4", "Gaussian blur sigma 1/2 px.", "Out-of-focus lens, haze, dust or water on the camera window."],
    ["7.5", "JPEG re-compression at quality 50/20/10.", "Bandwidth-limited transmission from a remote "
     "site; compression artefacts."],
    ["7.6", "Occlusion: grey out bottom 20% (stand), top-right 56x56 corner, and both.",
     "Direct test of the Grad-CAM hot spots (Step 6)."],
    ["7.7", "Controls (" + C % "--controls-only" + "): same-size 56x56 grey boxes at top-left, "
     "bottom-left and top-centre, and one over the panel centre.",
     "A grey box might confuse the model wherever it is placed. Background boxes separate 'this "
     "region matters' from 'any grey box hurts'; the panel-centre box tests reliance on the panel."],
    ["7.8", "Debug run on a random 500-image subset (" + C % "--limit 500" + ") before the full run.",
     "Checks the pipeline cheaply; its numbers are not used in the report."],
    ["7.9", "Metrics per condition: MAE, RMSE, " + R2 + ", bias, mean absolute prediction shift vs "
     "clean, MAE change vs clean.",
     "Bias gives the direction of the change; the shift measures how much predictions move even "
     "where the error does not change much."],
], [0.9 * cm, 7.6 * cm, 8.5 * cm], small=True))

# 3 method
s.append(Paragraph("3. Method", H1))
s.append(Paragraph("3.1 Commands", H2))
s.append(Preformatted(
    ".\\.venv\\Scripts\\Activate.ps1\n"
    "python robustness.py --limit 500      # debug run, 500 random test images (seed 0)\n"
    "python robustness.py                  # full grouped test set, 18 conditions\n"
    "python robustness.py --controls-only  # clean + 3 control boxes + panel-centre box", CODE))
s.append(Paragraph("3.2 Pipeline", H2))
s.extend(bullets([
    "Model: " + C % "build_model(pretrained=False)" + " + " +
    C % "outputs/checkpoints/resnet18_grouped_best.pt" + " (the Step 4 best checkpoint), eval mode, "
    "GPU, bf16 autocast exactly as in the Step 4 evaluation, predictions clamped to [0, 1].",
    "Images: the grouped test rows of " + C % "splits/splits.csv" + "; each image is loaded as RGB, "
    "resized to 224x224 and kept as uint8 (no normalisation, no augmentation).",
    "For every condition: convert to float in [0, 1] -&gt; <b>apply the perturbation</b> -&gt; "
    "ImageNet normalisation -&gt; model. Perturbing <i>before</i> normalisation means every "
    "perturbation acts on real pixel values, like a change in the camera or transmission.",
    "Same order of operations as " + C % "data.py" + " (resize, then tensor, then normalise), so the "
    "clean condition must reproduce Step 4 exactly; it does (MAE " + f4(clean.MAE) + ", " + R2 + " " +
    f"{clean.R2:.4f}" + ", check " + ok(clean_ok) + ").",
    "Randomness: only the noise conditions are random. A torch generator seeded with 42 is reset "
    "before every condition, so each run is reproducible. Only one noise seed was used.",
]))
s.append(Paragraph("3.3 Perturbation definitions (x = image in [0, 1], 224x224)", H2))
s.append(table([
    ["Family", "Definition in robustness.py", "Levels"],
    ["Brightness", "x' = clip(f x, 0, 1)", "f = 0.6, 0.8, 1.2, 1.4"],
    ["Contrast", "x' = clip(f (x - m) + m, 0, 1), m = mean of the whole image", "f = 0.6, 1.4"],
    ["Gaussian noise", "x' = clip(x + N(0, sigma<super>2</super>), 0, 1), independent per pixel and "
     "channel", "sigma = 0.02, 0.05, 0.10 (pixel units on [0, 1], i.e. about 5, 13, 26 grey levels)"],
    ["Gaussian blur", "torchvision gaussian_blur, kernel size 2 round(3 sigma) + 1", "sigma = 1 px "
     "(kernel 7), 2 px (kernel 13)"],
    ["JPEG", "round to uint8, save with PIL at quality q, decode", "q = 50, 20, 10"],
    ["Occlusion", "set a rectangle to 0.5 (mid grey) in all channels", "see Section 3.4"],
], [2.6 * cm, 9 * cm, 5.4 * cm], small=True))
s.append(Paragraph("3.4 Occlusion regions (pixel rows/cols of the 224x224 input)", H2))
s.append(table([
    ["Region", "Rows", "Cols", "Area", "Purpose"],
    ["Bottom 20% (stand/base)", "179-223", "0-223", "20.1%", "Grad-CAM hot spot 1 (hypothesis H1)"],
    ["Top-right corner", "0-55", "168-223", "6.25%", "Grad-CAM hot spot 2 (hypothesis H2)"],
    ["Both", "union", "", "26.3%", "Combined effect"],
    ["Control: top-left", "0-55", "0-55", "6.25%", "Background, same box size"],
    ["Control: bottom-left", "168-223", "0-55", "6.25%", "Background near the stand, same size"],
    ["Control: top-centre", "0-55", "84-139", "6.25%", "Upper image edge, overlaps the panel's top edge"],
    ["Panel centre", "84-139", "84-139", "6.25%", "Panel interior (hypothesis H3: panel matters)"],
], [4 * cm, 1.8 * cm, 1.8 * cm, 1.4 * cm, 8 * cm], small=True))
s.append(fig(FIG_OCC, 15,
             f"Figure 1. Occlusion regions drawn on one grouped test frame ({ex.day[-2:]} Jun "
             f"{ex.timestamp:%H:%M}, true L {ex.L:.3f}); solid = hypothesis regions, dashed = controls. "
             "Right: the grey fill as the model receives it. Made for this document."))
s.append(para(
    "Figure 1 shows where the boxes fall (the camera is fixed, so the layout is the same in every "
    "frame). The bottom strip covers the stand, the pipe and the gravel, plus the lower frame of the "
    "panel on the right. The top-right box mostly covers the neighbouring panel and the ground behind "
    "it, but also a small part of the main panel's top-right corner and frame. The top-centre control "
    "covers part of the upper panel surface and its top edge. The top-left control covers a "
    "background panel and ground; the bottom-left control covers the pipe and part of the stand area "
    "(so it is also a second, smaller stand test). The panel-centre box lies entirely on the panel "
    "surface and, in heavily soiled frames, overlaps the left edge of the deposits."))
s.append(fig(FIG_GAL, 17,
             "Figure 2. The same test frame under every photometric/noise/blur/JPEG condition "
             "(perturbation functions of robustness.py applied on the CPU; the noise sample is "
             "illustrative). Titles: MAE over the full grouped test set for that condition."))

# 4 results photometric
s.append(Paragraph("4. Results: photometric, noise, blur and compression", H1))
names = [c for c in res.condition if not c.startswith("occlude")]
rows, shade = res_table(R, names)
s.append(table(rows, W7, small=True, shade_rows=shade))
s.append(Paragraph("Source: outputs/eval/robustness_grouped_test.csv. Red: MAE more than 25% above "
                   "clean; green: MAE below clean. Shift = mean |prediction under perturbation - clean "
                   "prediction| per image.", CAP))
s.append(fig(FIG_SEV, 17, "Figure 3. MAE and bias against perturbation strength (plotted from the CSV). "
             "Grey dashed: clean MAE; red dotted: mean-predictor baseline MAE (0.2100)."))
s.append(fig(EV / "robustness_grouped_test.png", 13,
             "Figure 4. MAE for every condition including occlusion (outputs/eval/"
             "robustness_grouped_test.png); red dashed line = clean MAE."))
s.append(Paragraph("4.1 Findings by family", H2))
s.extend(bullets([
    f"<b>Brightness:</b> x0.8 and x1.2 change MAE by {rel('brightness x0.8'):+.1f}% and "
    f"{rel('brightness x1.2'):+.1f}%; x1.4 by {rel('brightness x1.4'):+.1f}%. Bias rises steadily "
    f"with brightness ({R.loc['brightness x0.6', 'bias(pred-true)']:+.4f} at x0.6 to "
    f"{R.loc['brightness x1.4', 'bias(pred-true)']:+.4f} at x1.4): a brighter image looks a little "
    "dirtier to the model (white deposits are bright). The training jitter (brightness +-15%) covers "
    "x0.8-1.2 only partly, yet the model also tolerates x0.6.",
    f"<b>x0.6 brightness gives a slightly lower MAE than clean</b> ({f4(R.loc['brightness x0.6', 'MAE'])}). "
    f"This is not a real improvement: darkening lowers all predictions, which cancels the model's "
    f"positive bias on this test set (clean bias {clean['bias(pred-true)']:+.4f}, x0.6 bias "
    f"{R.loc['brightness x0.6', 'bias(pred-true)']:+.4f}). Predictions still moved by "
    f"{f4(R.loc['brightness x0.6', 'mean|pred shift|'])} on average.",
    f"<b>Contrast:</b> x0.6 and x1.4 change MAE by {rel('contrast x0.6'):+.1f}% and "
    f"{rel('contrast x1.4'):+.1f}%; higher contrast raises the bias slightly (same direction as brightness).",
    f"<b>Noise:</b> sigma 0.02 is harmless ({rel('noise sigma 0.02'):+.1f}%), sigma 0.05 gives "
    f"{rel('noise sigma 0.05'):+.1f}%, and sigma 0.10 breaks the model: MAE "
    f"{f4(R.loc['noise sigma 0.10', 'MAE'])}, {R2} {R.loc['noise sigma 0.10', 'R2']:.4f}, bias "
    f"{R.loc['noise sigma 0.10', 'bias(pred-true)']:+.4f}. The error is almost entirely a "
    f"systematic upward shift (mean shift {f4(R.loc['noise sigma 0.10', 'mean|pred shift|'])}): "
    "speckle noise looks like fine dirt to the network, so clean panels are predicted as soiled. "
    f"With a negative {R2} and MAE above the baseline ({BASE_MAE:.4f}), the model is then worse than "
    "predicting a constant.",
    f"<b>Blur:</b> sigma 1 is tolerated ({rel('blur sigma 1'):+.1f}%) but already moves predictions "
    f"by {f4(R.loc['blur sigma 1', 'mean|pred shift|'])} with a negative bias "
    f"({R.loc['blur sigma 1', 'bias(pred-true)']:+.4f}); sigma 2 gives {rel('blur sigma 2'):+.1f}% "
    f"and {R2} {R.loc['blur sigma 2', 'R2']:.3f} with bias {R.loc['blur sigma 2', 'bias(pred-true)']:+.4f}. "
    "Blur smooths away the edges and texture of deposits, so panels look cleaner. Blur is the only "
    "family that pushes predictions <b>down</b>.",
    f"<b>JPEG:</b> q50 is harmless ({rel('JPEG q50'):+.1f}%); q20 ({rel('JPEG q20'):+.1f}%) and q10 "
    f"({rel('JPEG q10'):+.1f}%) push predictions up (bias {R.loc['JPEG q20', 'bias(pred-true)']:+.4f} and "
    f"{R.loc['JPEG q10', 'bias(pred-true)']:+.4f}). Block artefacts add texture that resembles soiling. "
    "The source frames are already compressed, so this is a second compression on top.",
    "<b>Small 'improvements'</b> (green rows) under brightness x0.6, contrast x0.6 and noise 0.02 are "
    "all at most 4% and come with a lower bias; they reflect the cancellation of the positive bias, "
    "not a better model, and should not be over-interpreted.",
    f"<b>RMSE vs MAE:</b> under mild conditions RMSE barely moves (0.129-0.134 vs "
    f"{f4(clean.RMSE)}), because the RMSE is dominated by the few worst frames (true ~0.9, predicted "
    "~0.1, Step 5), which no perturbation fixes.",
]))

# 5 occlusion
s.append(Paragraph("5. Results: occlusion and controls", H1))
occ = ["occlude bottom 20%", "occlude top-right corner", "occlude both regions"]
rows, shade = res_table(R, ["clean (no change)"] + occ)
for c in ctl.condition[1:]:
    r = CT.loc[c]
    d = 100 * (r.MAE / clean.MAE - 1)
    rows.append([c, f4(r.MAE), f4(r.RMSE), f"{r.R2:.4f}", f"{r['bias(pred-true)']:+.4f}",
                 f4(r["mean|pred shift|"]), f"{d:+.1f}%"])
    if d > 25:
        shade[len(rows) - 1] = "#FCE8E6"
    elif d < 0:
        shade[len(rows) - 1] = "#E8F4EA"
s.append(table(rows, W7, small=True, shade_rows=shade))
s.append(Paragraph("Sources: outputs/eval/robustness_grouped_test.csv (first four rows) and "
                   "robustness_grouped_test_controls.csv (last four rows; its clean row is identical).",
                   CAP))
s.append(fig(EV / "robustness_grouped_test_controls.png", 12,
             "Figure 5. MAE for the control boxes and the panel-centre box (outputs/eval/"
             "robustness_grouped_test_controls.png)."))
s.append(Paragraph("5.1 Findings", H2))
s.extend(bullets([
    f"<b>H1 (stand) rejected.</b> Hiding the bottom 20% gives MAE {f4(R.loc['occlude bottom 20%', 'MAE'])} "
    f"({rel('occlude bottom 20%'):+.1f}%) and {R2} {R.loc['occlude bottom 20%', 'R2']:.4f}, slightly "
    f"<i>better</i> than clean, even though it hides a fifth of the image. Predictions move by "
    f"{f4(R.loc['occlude bottom 20%', 'mean|pred shift|'])} on average and downwards (bias "
    f"{R.loc['occlude bottom 20%', 'bias(pred-true)']:+.4f}), so the stand region did add a little "
    "to the predicted loss (consistent with the positive Grad-CAM evidence there), but the model "
    "does not need it to be accurate. As with brightness x0.6, the small gain comes from cancelling "
    "the positive bias.",
    f"<b>Controls: a grey box alone is harmless.</b> Top-left: MAE "
    f"{f4(CT.loc['control: top-left corner 56x56', 'MAE'])}, shift "
    f"{f4(CT.loc['control: top-left corner 56x56', 'mean|pred shift|'])}; bottom-left: "
    f"{f4(CT.loc['control: bottom-left corner 56x56', 'MAE'])}, shift "
    f"{f4(CT.loc['control: bottom-left corner 56x56', 'mean|pred shift|'])}. So any large effect of "
    "the other boxes comes from <i>what</i> they hide, not from the grey box itself.",
    f"<b>H3 (panel matters) supported.</b> Hiding a 56x56 box in the panel centre (6.25% of the "
    f"image, the same size as the controls) gives MAE {f4(CT.loc['occlude panel centre 56x56', 'MAE'])} "
    f"({rel('occlude panel centre 56x56', CT):+.0f}%, about 2x), RMSE "
    f"{f4(CT.loc['occlude panel centre 56x56', 'RMSE'])}, {R2} "
    f"{CT.loc['occlude panel centre 56x56', 'R2']:.4f}. Predictions depend mainly on the panel area. "
    f"The bias jumps to {CT.loc['occlude panel centre 56x56', 'bias(pred-true)']:+.4f}: a flat grey "
    "patch on a dark blue panel can itself look like a deposit, so this shows that the panel region "
    "is sensitive, not exactly which features are used.",
    f"<b>H2 (top-right corner) supported, cause unclear.</b> MAE "
    f"{f4(R.loc['occlude top-right corner', 'MAE'])} ({rel('occlude top-right corner'):+.0f}%), {R2} "
    f"{R.loc['occlude top-right corner', 'R2']:.4f}, bias {R.loc['occlude top-right corner', 'bias(pred-true)']:+.4f}. "
    "Because the same-size background controls have no effect, this region genuinely matters. "
    "However, the box covers both the neighbouring panel and the main panel's own corner and frame "
    "(Figure 1), so the experiment cannot say whether the model uses the neighbouring panel (a "
    "shortcut) or the corner of the main panel (legitimate). It may also be that grey on the panel "
    "corner looks like soiling, as for the centre box (the bias direction is the same).",
    f"<b>Top-centre control:</b> MAE {f4(CT.loc['control: top-centre 56x56', 'MAE'])} "
    f"({rel('control: top-centre 56x56', CT):+.0f}%). This box overlaps the panel's upper edge, so a "
    "moderate effect fits the picture: the more of the panel a box covers, the larger the damage.",
    f"<b>Both regions:</b> MAE {f4(R.loc['occlude both regions', 'MAE'])}, about the same as top-right "
    f"alone, but a better {R2} ({R.loc['occlude both regions', 'R2']:.4f} vs "
    f"{R.loc['occlude top-right corner', 'R2']:.4f}) and smaller bias "
    f"({R.loc['occlude both regions', 'bias(pred-true)']:+.4f}): the downward push from hiding the "
    "stand partly cancels the upward push from hiding the corner.",
]))
s.append(Paragraph("5.2 Ordering of the effect by region", H2))
order = sorted([("bottom 20% (stand)", R.loc["occlude bottom 20%"]),
                ("control: top-left", CT.loc["control: top-left corner 56x56"]),
                ("control: bottom-left", CT.loc["control: bottom-left corner 56x56"]),
                ("control: top-centre", CT.loc["control: top-centre 56x56"]),
                ("top-right corner", R.loc["occlude top-right corner"]),
                ("panel centre", CT.loc["occlude panel centre 56x56"])],
               key=lambda t: t[1]["mean|pred shift|"])
s.append(table([["Region hidden", "Mean |prediction shift|", "MAE", "On the panel?"]] +
               [[n, f4(r["mean|pred shift|"]), f4(r.MAE),
                 {"bottom 20% (stand)": "lowest panel strip only", "control: top-left": "no",
                  "control: bottom-left": "no", "control: top-centre": "upper edge",
                  "top-right corner": "corner + frame", "panel centre": "yes, fully"}[n]]
                for n, r in order],
               [5 * cm, 4 * cm, 2.6 * cm, 5.4 * cm], small=True))
s.append(para("Sorted by how much the predictions move. The ranking follows how much of the panel surface "
              "each box covers: pure background boxes barely move predictions, boxes on the panel move "
              "them a lot. This is the quantitative answer to the Step 6 question."))

# 6 debug run
s.append(Paragraph("6. Debug run (500 images) vs full run", H1))
s.append(para(
    "Before the full run, " + C % "--limit 500" + " evaluated a random 500-image subset of the grouped "
    "test set (seed 0) in about a minute. Its numbers are kept in " +
    C % "robustness_grouped_test_lim500.csv/.png" + " for traceability only."))
cmp_rows = [["Condition", "MAE (500)", "MAE (full)", "MAE change vs clean (500)",
             "MAE change vs clean (full)"]]
for c in ["clean (no change)", "brightness x1.4", "noise sigma 0.10", "blur sigma 2", "JPEG q10",
          "occlude bottom 20%", "occlude top-right corner"]:
    cmp_rows.append([c, f4(DB.loc[c, "MAE"]), f4(R.loc[c, "MAE"]),
                     "-" if c.startswith("clean") else
                     f"{100 * (DB.loc[c, 'MAE'] / DB.loc['clean (no change)', 'MAE'] - 1):+.1f}%",
                     "-" if c.startswith("clean") else f"{rel(c):+.1f}%"])
s.append(table(cmp_rows, [5 * cm, 2.4 * cm, 2.4 * cm, 3.6 * cm, 3.6 * cm], small=True))
s.append(para(
    f"The subset has a slightly higher clean MAE ({f4(DB.loc['clean (no change)', 'MAE'])}) and lower "
    f"{R2} ({DB.loc['clean (no change)', 'R2']:.4f}) than the full set, as expected from sampling, "
    "but every condition changes the error in the same direction and by a similar relative amount. "
    "This also shows that the conclusions are not sensitive to which test frames are used. "
    "<b>Only the full-run numbers are reported.</b>"))

# 7 interpretation
s.append(Paragraph("7. Interpretation", H1))
s.extend(bullets([
    "<b>Operating range.</b> The model is usable under moderate changes of exposure and contrast, "
    "light sensor noise, slight defocus and moderate compression (q50). These are the variations a "
    "fixed outdoor camera is most likely to see from day to day.",
    "<b>Failure modes are systematic, not random.</b> Every severe degradation has a clear sign: "
    "texture-adding perturbations (noise, JPEG artefacts) inflate the predicted loss, texture-removing "
    "ones (blur) deflate it. The model has learned that <i>fine texture/bright speckle on the panel "
    "= soiling</i>, which is physically sensible for dust but means image-quality problems are "
    "misread as dirt or as cleanliness.",
    "<b>Practical consequence.</b> In deployment, a noisy or heavily compressed camera would cause "
    "false 'needs cleaning' alarms, and a dirty or out-of-focus camera window would hide real soiling. "
    "Image-quality checks or training with noise/blur/JPEG augmentation would be needed.",
    "<b>Tied to the camera.</b> The frames are small (192x192) and already compressed (~17 KB); the "
    "model has seen only this one camera and compression setup, so results on another camera may "
    "differ more than these synthetic perturbations suggest.",
    "<b>Grad-CAM revisited.</b> Grad-CAM's stand hot spot was not a real dependence; the panel "
    "interior is what the predictions rely on most. The top-right dependence remains unexplained; "
    "the survey's proposal to crop/standardise the panel would remove the neighbouring panel and the "
    "stand from the input and make the model's evidence easier to interpret.",
]))

# 8 limitations
s.append(Paragraph("8. Caveats and limitations", H1))
s.extend(bullets([
    "<b>One noise seed</b> (42): the noise rows are a single random draw; with 6,733 images the "
    "average is stable, but no spread over seeds was measured.",
    "<b>Synthetic perturbations</b> applied to the already-resized 224x224 image; real camera changes "
    "(sensor, lens, white balance, viewing angle) combine several effects and were not tested.",
    "<b>Flat grey occluder is a coarse probe</b>: it removes information but also adds an artificial "
    "uniform patch that can look like a deposit (positive bias for panel boxes). Other fills (blur, "
    "mean colour, inpainting) were not tried.",
    "<b>Box placement is fixed in pixel coordinates</b>; it suits this dataset because the camera is "
    "fixed, but the boxes do not separate the neighbouring panel from the main panel corner.",
    "<b>Only overall metrics were saved</b> (no per-image predictions, per-day or per-L-range "
    "breakdowns per condition), so it is not known whether, for example, noise hurts clean panels "
    "more than soiled ones (the positive bias suggests so).",
    "<b>Two test days, one model, one seed, one site/camera</b>, as in all previous steps. No "
    "retraining with augmentation was attempted, so we only measure sensitivity, not how to fix it.",
    "<b>Test set use:</b> perturbations were chosen in advance and no model or setting was changed "
    "based on these results, so the grouped test set remains a clean held-out estimate for the "
    "headline numbers.",
]))

# 9 outputs
s.append(Paragraph("9. Outputs produced", H1))
s.append(table([
    ["File", "Content"],
    ["robustness.py", "Step 7 script (perturbations, occlusions, controls; " + C % "--limit" + ", " +
     C % "--controls-only" + ", " + C % "--strategy" + ")."],
    ["outputs/eval/robustness_grouped_test.csv / .png",
     "18 conditions (clean, 14 perturbations, 3 occlusions): MAE, RMSE, " + R2 +
     ", bias, mean |pred shift|, MAE change vs clean; bar chart (Figure 4)."],
    ["outputs/eval/robustness_grouped_test_controls.csv / .png",
     "Clean + 3 control boxes + panel-centre box (Figure 5)."],
    ["outputs/eval/robustness_grouped_test_lim500.csv / .png", "Debug run on 500 images (not reported)."],
    ["docs/step7_figs/", "Figures 1-3 of this document (made by docs/build_step7_pdf.py)."],
], [7.4 * cm, 9.6 * cm], small=True))

# 10 next & viva
s.append(Paragraph("10. Next steps", H1))
s.extend(bullets([
    "Write the report (Abstract, Introduction, Survey, Dataset, Method, Experiments/Results, "
    "Conclusion, References, Contributions) and the slides; rehearse for the viva. Report hardcopy "
    "due Monday 23 Nov 2026.",
    "Optional if time: inference time and parameter count; sensitivity analysis excluding I &lt; 0.1; "
    "EfficientNet-B0 comparison. Future work: panel cropping/standardisation, training with "
    "noise/blur/JPEG augmentation, several seeds.",
]))
s.append(KeepTogether([Paragraph("10.1 Viva points from this step", H2), table([
    ["Question", "Answer grounded in Step 7"],
    ["How robust is the model?",
     "MAE changes at most about 11% under brightness/contrast x0.6-1.4, noise sigma 0.02, blur "
     "sigma 1, JPEG q50; strong noise (sigma 0.10), blur sigma 2 and JPEG q20/q10 degrade it badly."],
    ["Why does noise increase the predicted loss?",
     "Speckle looks like fine dirt; the model associates bright fine texture with soiling. Blur "
     "removes texture and lowers predictions."],
    ["Was the model retrained for these tests?", "No; the trained grouped model is evaluated on "
     "perturbed copies of the grouped test set only."],
    ["Does the model use the stand (Grad-CAM hot spot)?",
     f"No: hiding the bottom 20% gives MAE {f4(R.loc['occlude bottom 20%', 'MAE'])} vs "
     f"{f4(clean.MAE)} clean."],
    ["How do you know the grey box itself isn't the problem?",
     "Same-size boxes in background corners change nothing (MAE 0.0869 / 0.0844), while the same "
     "box on the panel centre doubles the error (0.1725)."],
    ["Why is MAE sometimes better after a perturbation?",
     "The model over-predicts on this test set (bias +0.017); perturbations that lower predictions "
     "cancel part of that bias. Not a real improvement."],
], [5.2 * cm, 11.8 * cm], small=True)]))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Step 7: Robustness and Occlusion Tests",
                        author="IDSE Solar Soiling Team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("clean row matches Step 4:", clean_ok, "| controls clean identical:", ctl_clean_ok)
print("example frame:", ex.filename, "| test n:", n_test)
print("Wrote", PDF)
