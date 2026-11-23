"""Build docs/Step3_Data_Pipeline_and_Baseline.pdf (Step 3 documentation).

Usage (from the project folder, venv active):
    python -m pip install reportlab
    python docs/build_step3_pdf.py

Numbers that are cheap to recompute are recomputed here (dataset sizes, one batch check,
mean-predictor baseline incl. per-range MAE, software versions). The loading-speed figures
are quoted from the recorded run of data.py (re-timing them here would give different
numbers on every run). Raw images are only opened read-only.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torchvision.transforms.functional as TF
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
import os
os.chdir(ROOT)                      # data.py uses the relative path splits/splits.csv
import data                         # noqa: E402  (shared pipeline, unchanged)

FIG_DIR = ROOT / "docs" / "step3_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF = ROOT / "docs" / "Step3_Data_Pipeline_and_Baseline.pdf"
SETS = ("train", "val", "test")
FROZEN = {("random", "train"): 32027, ("random", "val"): 6863, ("random", "test"): 6864,
          ("grouped", "train"): 32158, ("grouped", "val"): 6863, ("grouped", "test"): 6733}

# ---------------------------------------------------------------- environment
cuda_ok = torch.cuda.is_available()
gpu = torch.cuda.get_device_name(0) if cuda_ok else "none"
vers = {"Python": sys.version.split()[0], "torch": torch.__version__,
        "torchvision": torchvision.__version__, "CUDA (torch build)": str(torch.version.cuda),
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "Pillow": PIL.__version__, "numpy": np.__version__}

# ---------------------------------------------------------------- dataset checks
sizes = {}
for st in ("random", "grouped"):
    for sb in SETS:
        sizes[(st, sb)] = len(data.SoilingDataset(st, sb))
sizes_ok = all(sizes[k] == FROZEN[k] for k in FROZEN)

torch.manual_seed(42)
loader = data.make_loader("grouped", "train", batch_size=16, num_workers=0)
xb, yb, nb = next(iter(loader))
batch = {"shape": tuple(xb.shape), "x_dtype": str(xb.dtype), "y_dtype": str(yb.dtype),
         "y_min": float(yb.min()), "y_max": float(yb.max()),
         "x_min": float(xb.min()), "x_max": float(xb.max()), "n_names": len(nb)}
batch_ok = (batch["shape"][1:] == (3, 224, 224) and yb.dtype == torch.float32
            and 0 <= batch["y_min"] and batch["y_max"] <= 1)
raw = PILImage.open(data.DATA_ROOT / nb[0])
raw_info = f"{raw.size[0]}x{raw.size[1]}, mode {raw.mode}"

# ---------------------------------------------------------------- jitter-range figure
df = pd.read_csv(ROOT / "splits" / "splits.csv")
tr = df[df["grouped_split"] == "train"]
ex = tr.iloc[(tr["L"] - 0.73).abs().argsort().iloc[0]]
img = PILImage.open(data.DATA_ROOT / ex["filename"]).convert("RGB").resize((224, 224))
variants = [("original", img),
            ("brightness x0.85", TF.adjust_brightness(img, 0.85)),
            ("brightness x1.15", TF.adjust_brightness(img, 1.15)),
            ("contrast x0.85", TF.adjust_contrast(img, 0.85)),
            ("contrast x1.15", TF.adjust_contrast(img, 1.15))]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
for a, (t, im) in zip(axes, variants):
    a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
fig.suptitle(f"Extremes of the training jitter on one train image (L = {ex['L']:.3f})", fontsize=11)
plt.tight_layout()
FIG_JIT = FIG_DIR / "jitter_extremes.png"
plt.savefig(FIG_JIT, dpi=120); plt.close()

# ---------------------------------------------------------------- baseline (recomputed)
RANGES = [(0.0, 0.15, "low (L < 0.15)"), (0.15, 0.5, "mid (0.15-0.5)"), (0.5, 1.01, "high (L >= 0.5)")]


def metrics(y, p):
    e = y - p
    return (np.abs(e).mean(), np.sqrt((e ** 2).mean()),
            1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


base, per_range, train_mean = [], [], {}
for st in ("random", "grouped"):
    col = f"{st}_split"
    m = df.loc[df[col] == "train", "L"].mean()
    train_mean[st] = m
    for sb in ("val", "test"):
        y = df.loc[df[col] == sb, "L"].to_numpy()
        mae, rmse, r2 = metrics(y, np.full_like(y, m))
        base.append((st, sb, len(y), y.mean(), mae, rmse, r2))
        for lo, hi, name in RANGES:
            k = (y >= lo) & (y < hi)
            per_range.append((st, sb, name, int(k.sum()), 100 * k.mean(),
                              np.abs(y[k] - m).mean()))
saved = pd.read_csv(ROOT / "baseline_results.csv")
base_ok = all(
    np.isclose(saved.loc[(saved.strategy == st) & (saved.set == sb), ["MAE", "RMSE", "R2"]]
               .to_numpy().ravel(), np.round([mae, rmse, r2], 4), atol=1e-4).all()
    for st, sb, _, _, mae, rmse, r2 in base)

# ---------------------------------------------------------------- styles (same as Steps 1-2)
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


def p(t):
    return Paragraph(t, B)


def bullets(items):
    return [Paragraph(t, BL, bulletText="\u2022") for t in items]


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


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "IDSE project - Solar panel soiling power-loss regression - Step 3 documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------- content
s = []
s.append(Paragraph("Step 3: Data Pipeline, Environment and Baseline", ss["Title"]))
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
s.append(Paragraph("Document date: 19 Nov 2026. Presentation: 21 Nov 2026. "
                   "Status: Step 3 <b>done</b>; Step 4 (ResNet-18 training) is next.", CAP))

# 1
s.append(Paragraph("1. Purpose of Step 3 and where it fits", H1))
s.append(p(
    "Step 1 audited the 45,754 images and fixed the target (L, power loss as a fraction 0-1). "
    "Step 2 froze two splits (random and grouped-by-day). Step 3 builds everything that sits "
    "between those split files and the neural network:"))
s.extend(bullets([
    "a working <b>GPU environment</b> for PyTorch on the team's laptop;",
    "one <b>shared data pipeline</b> (" + C % "data.py" + ") so that every teammate loads exactly "
    "the same images, with exactly the same preprocessing, from the frozen splits;",
    "<b>sanity checks</b> that the tensors, labels and augmentation are correct before any GPU "
    "time is spent on training;",
    "a <b>mean-predictor baseline</b> (" + C % "baseline.py" + "): the score a model gets "
    "without looking at the image at all. The CNN in Step 4 is only useful if it clearly beats "
    "these numbers.",
]))
s.append(p("Nothing in Step 3 changes the raw images or the split files; both are only read."))

# 2
s.append(Paragraph("2. What we did and why", H1))
s.append(table([
    ["#", "Action", "Why"],
    ["3.1", "Created a Python 3.12 virtual environment (" + C % ".venv" + ") and installed the "
            "CUDA 12.8 build of PyTorch plus pandas, matplotlib, Pillow.",
     "The default Python on the PC (3.15) has no PyTorch wheels, and the RTX 50-series GPU "
     "needs the cu128 build. A venv keeps the project isolated and reproducible."],
    ["3.2", "Wrote " + C % "SoilingDataset" + ": reads " + C % "splits/splits.csv" + ", selects "
            "one strategy and subset, opens the image, returns (image tensor, L, filename).",
     "Single source of truth for data loading; filenames are returned so predictions can be "
     "saved per image (needed later for error analysis and Grad-CAM example selection)."],
    ["3.3", "Fixed the preprocessing: resize 192 to 224, tensor, ImageNet normalisation; mild "
            "brightness/contrast jitter on train only.",
     "Matches what the ImageNet-pretrained ResNet-18 expects; augmentation adds lighting "
     "variety without changing what the soiling looks like."],
    ["3.4", "Loaded one batch and checked shape, dtype and target range; saved a visual grid "
            "of train and test images (" + C % "sanity_batch.png" + ").",
     "Catches wrong paths, wrong label units, wrong normalisation or destructive augmentation "
     "before training."],
    ["3.5", "Computed the mean-predictor baseline on val and test for both split strategies, "
            "overall and per L range.",
     "Gives the minimum bar for the CNN, and per-range numbers because L is strongly skewed."],
    ["3.6", "Timed data loading with 0 and 4 worker processes.",
     "Decides whether loading or the GPU is the bottleneck, and which " + C % "num_workers" +
     " to use in training."],
], [0.9 * cm, 7.6 * cm, 8.5 * cm]))

# 3 environment
s.append(Paragraph("3. Environment (3.1)", H1))
s.append(table([["Item", "Value"]] +
               [["GPU", gpu + " (8 GB), NVIDIA driver 591.59"],
                ["torch.cuda.is_available()", str(cuda_ok)]] +
               [[k, v] for k, v in vers.items()] +
               [["Extra (documentation only)", "reportlab (to build these PDFs)"],
                ["OS / shell", "Windows 11, PowerShell"]],
               [6 * cm, 11 * cm]))
s.append(Spacer(1, 4))
s.extend(bullets([
    "<b>Why not the system Python:</b> the PC's default " + C % "python" + " is 3.15, for which "
    "PyTorch publishes no wheels, so " + C % "pip install torch" + " fails. Python 3.12 is "
    "supported, so a 3.12 venv was created inside the project folder.",
    "<b>Why cu128:</b> the RTX 5060 (Blackwell, RTX 50-series) is only supported by PyTorch "
    "builds compiled against CUDA 12.8. The cu121 build installs but cannot run kernels on this "
    "GPU. The values above are read from the venv when this document is built.",
    "Every command below assumes the venv is active.",
]))
s.append(Preformatted(
    "cd C:\\Users\\manie\\OneDrive\\Desktop\\solar-soiling\n"
    "py -3.12 -m venv .venv                       # one-time\n"
    ".\\.venv\\Scripts\\Activate.ps1                 # every session\n"
    "python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
    "python -m pip install pandas matplotlib pillow\n"
    "python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"",
    CODE))
s.append(p("Teammates on a different GPU (or Colab) install the PyTorch build that matches their "
           "hardware; the rest of the pipeline is unchanged."))

# 4 pipeline
s.append(Paragraph("4. Shared data pipeline: data.py (3.2-3.3)", H1))
s.append(Paragraph("4.1 Configuration (top of data.py)", H2))
s.append(table([
    ["Setting", "Value", "Note"],
    [C % "DATA_ROOT", "...\\Solar_Panel_Soiling_Image_dataset\\PanelImages",
     "The only line a teammate edits, or set the environment variable " +
     C % "SOLAR_DATA_ROOT" + " instead. It must point to the <b>PanelImages</b> folder itself, "
     "because the image path is built as " + C % "DATA_ROOT / filename" + "."],
    [C % "SPLITS_CSV", C % "splits/splits.csv", "Relative path: run scripts from the project folder."],
    [C % "IMG_SIZE", "224", "Input size of ImageNet-pretrained ResNet-18."],
    [C % "MEAN, STD", "[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]",
     "ImageNet channel statistics used when the backbone was pretrained."],
], [2.8 * cm, 5.4 * cm, 8.8 * cm]))
s.append(Paragraph("4.2 SoilingDataset and make_loader", H2))
s.extend(bullets([
    C % "SoilingDataset(strategy, subset)" + ": " + C % "strategy" + " is 'random' or 'grouped', "
    + C % "subset" + " is 'train', 'val' or 'test'. It filters " + C % "splits.csv" + " on the "
    "column " + C % "&lt;strategy&gt;_split" + ", so it can only ever use the frozen assignment.",
    "Each item is opened with Pillow and converted to RGB, transformed, and returned as "
    "<b>(image tensor [3, 224, 224], L as float32 fraction, filename)</b>. The target stays a "
    "fraction; percent is only used when reporting.",
    "Augmentation is on for the train subset by default and off for val/test (overridable with "
    + C % "train_aug" + ", e.g. to evaluate train images without jitter).",
    C % "make_loader(strategy, subset, batch_size=32, num_workers=4)" + ": shuffles only the "
    "train subset, uses pinned memory when a GPU is present, and keeps worker processes alive "
    "between epochs (" + C % "persistent_workers" + ") to avoid the Windows start-up cost every "
    "epoch.",
    "Split CSVs hold filenames only (no absolute paths), so the same files work on every "
    "teammate's machine.",
]))
s.append(Paragraph("4.3 Preprocessing and augmentation (fixed spec)", H2))
s.append(table([
    ["Order", "Operation", "Train", "Val/Test", "Why"],
    ["1", "Load, convert to RGB", "yes", "yes", "Images are 192x192 RGB JPEG (Step 1); convert is a "
                                               "safety net against grayscale/RGBA files."],
    ["2", "Resize to 224x224 (bilinear)", "yes", "yes", "ResNet-18 was pretrained at 224. Images are "
     "square, so no distortion and <b>no cropping</b>: the whole panel stays in view."],
    ["3", "ColorJitter(brightness 0.15, contrast 0.15)", "yes", "no",
     "Light changes over the day and between days. Random factors in [0.85, 1.15] teach the "
     "model to ignore exposure differences. Kept small because soiling itself is a brightness "
     "cue (white deposits); strong jitter could make clean and soiled panels look alike."],
    ["4", "ToTensor", "yes", "yes", "Pixel values to float in [0, 1], channel-first layout."],
    ["5", "Normalize(ImageNet mean/std)", "yes", "yes", "Inputs on the same scale as during "
     "pretraining, so pretrained filters respond as intended."],
], [1.2 * cm, 4.2 * cm, 1.2 * cm, 1.5 * cm, 8.9 * cm], small=True))
s.append(Spacer(1, 4))
s.append(p("<b>Deliberately not used:</b>"))
s.extend(bullets([
    "<b>Horizontal/vertical flips and rotations</b>: the camera is fixed, so the panel always "
    "appears in the same position and perspective. Flipped views never occur at test time and "
    "only add unrealistic variation.",
    "<b>Random / heavy crops</b>: soiling can be anywhere on the panel and L describes the "
    "<b>whole</b> panel. A crop could remove the soiled area while keeping the high label, "
    "which creates wrong training pairs.",
    "<b>Hue/saturation jitter, blur, noise</b>: they change how dust and rust look. Brightness, "
    "noise and JPEG perturbations are instead used later as a <b>robustness test</b> on the "
    "test set, not in training.",
    "<b>Any augmentation on val/test</b>: evaluation must be deterministic and comparable.",
    "<b>Irradiance I as an input</b>: the model sees only the image. I is a property of the "
    "measurement, not of the image, and the target is L.",
]))
s.append(KeepTogether([
    Image(str(FIG_JIT), width=17 * cm, height=17 * cm * 3.4 / 15),
    Paragraph("Figure 1. The strongest possible brightness and contrast changes applied by the "
              "training jitter (made by docs/build_step3_pdf.py). The deposits stay clearly "
              "visible in every case.", CAP)]))

# 5 checks
s.append(Paragraph("5. Sanity checks (3.4)", H1))
s.append(Preformatted("python data.py      # prints sizes, checks one batch, saves sanity_batch.png, "
                      "times loading", CODE))
rows = [["Strategy", "Set", "Images from SoilingDataset", "Frozen split size", "Match"]]
for (st, sb), n in sizes.items():
    rows.append([st, sb, f"{n:,}", f"{FROZEN[(st, sb)]:,}", ok(n == FROZEN[(st, sb)])])
s.append(table(rows, [2.4 * cm, 1.8 * cm, 4.6 * cm, 4.2 * cm, 4 * cm], small=True))
s.append(Spacer(1, 6))
s.append(table([
    ["Batch check (grouped train, batch 16, seed 42)", "Result"],
    ["Raw image on disk", raw_info],
    ["Image tensor shape", str(batch["shape"])],
    ["Image dtype / value range after normalisation",
     f"{batch['x_dtype']}, {batch['x_min']:.2f} to {batch['x_max']:.2f}"],
    ["Target dtype / range in this batch",
     f"{batch['y_dtype']}, {batch['y_min']:.3f} to {batch['y_max']:.3f} (fraction, inside [0, 1])"],
    ["Filenames returned", f"{batch['n_names']} (one per image)"],
    ["Shape, dtype and range assertions", ok(batch_ok)],
], [8 * cm, 9 * cm]))
s.append(p("Normalised pixel values are no longer in [0, 1]; roughly -2.1 to 2.6 is the expected "
           "range after ImageNet normalisation, which confirms that Normalize was applied once."))
s.append(KeepTogether([
    Image(str(ROOT / "sanity_batch.png"), width=17 * cm, height=17 * cm * 605 / 2200),
    Paragraph("Figure 2. sanity_batch.png. Top: 8 training images after the train transform "
              "(jitter on). Bottom: 8 evenly spaced grouped-test images (no augmentation). "
              "Images are un-normalised for display.", CAP)]))
s.extend(bullets([
    "The <b>whole panel</b> is visible in every image: resizing does not crop anything.",
    "The jitter is <b>mild</b>: jittered train images look natural and deposits are unchanged.",
    "Labels match the visual impression in most cases (L near 0 = clean panel; large white "
    "patches = high L).",
    "<b>Label noise is visible</b>: a panel with only a few rust-coloured spots has L = 0.891, "
    "while a panel with a large band of white deposits has L = 0.731. The image-to-loss relation "
    "is not purely about the visible area of dirt, so a very high R<super>2</super> should not "
    "be expected.",
]))

# 6 baseline
s.append(Paragraph("6. Mean-predictor baseline (3.5)", H1))
s.append(p(
    "The baseline predicts the <b>mean L of the training set</b> of the same strategy for every "
    "val/test image. It uses no image information, so any model that has learned something from "
    "the pictures must beat it. Using the training mean (not the test mean) is required: the "
    "test labels may not be used to make predictions."))
s.append(Preformatted(
    "python baseline.py      # writes baseline_results.csv (pandas + numpy only, no GPU)\n\n"
    "MAE  = mean(|y - p|)                       (average error, same unit as L)\n"
    "RMSE = sqrt(mean((y - p)^2))              (penalises large errors more)\n"
    "R2   = 1 - sum((y - p)^2) / sum((y - mean(y))^2)   (1 = perfect, 0 = as good as the set's own mean)",
    CODE))
rows = [["Strategy", "Set", "n", "Train mean L", "Set mean L", "MAE", "MAE (% pts)", "RMSE", "R2"]]
for st, sb, n, ym, mae, rmse, r2 in base:
    rows.append([st, sb, f"{n:,}", f4(train_mean[st]), f4(ym), f4(mae), f"{100 * mae:.2f}",
                 f4(rmse), f"{r2:+.4f}"])
s.append(table(rows, [1.8 * cm, 1.2 * cm, 1.5 * cm, 2.2 * cm, 2 * cm, 1.6 * cm, 2 * cm,
                      1.6 * cm, 1.6 * cm], small=True, shade_rows={3: "#F7F9FB", 4: "#F7F9FB"}))
s.append(Paragraph(f"Recomputed values agree with the saved baseline_results.csv: {ok(base_ok)}.",
                   CAP))
s.append(Spacer(1, 4))
s.extend(bullets([
    "<b>MAE of 0.21-0.25</b> means the baseline is wrong by 21-25 percentage points of power "
    "loss on average. This is the number to beat.",
    "<b>R<super>2</super> is about 0 on the random split</b>, as expected: train and test have "
    "the same mean, and predicting the mean is the definition of R<super>2</super> = 0.",
    f"<b>R<super>2</super> is negative on grouped val (-0.042)</b>: the grouped val days have a "
    f"lower mean L ({base[2][3]:.3f}) than the train days ({train_mean['grouped']:.3f}), so the "
    "training mean is a worse constant than val's own mean. This is the day-to-day label shift "
    "from Step 2, visible before any model is trained.",
    "<b>Grouped test MAE (0.210) is lower than random test MAE (0.246)</b>, not because grouped is "
    "easier for a real model, but because grouped test has a narrower L distribution "
    "(std 0.245 vs 0.282) with fewer near-0 and near-1 values, so a constant guess is closer on "
    "average. This is why the CNN must always be compared to the baseline <b>of the same split "
    "and set</b>, and why R<super>2</super> (which normalises by each set's spread) is reported "
    "next to MAE.",
]))
s.append(Paragraph("6.1 Baseline MAE per L range", H2))
s.append(p("L is strongly right-skewed (Step 1), so one overall MAE hides where errors occur. "
           "The same three ranges will be used for the CNN."))
rows = [["Strategy", "Set", "L range", "n", "Share of set", "Baseline MAE"]]
for st, sb, name, n, sh, mae in per_range:
    rows.append([st, sb, name, f"{n:,}", f"{sh:.1f}%", f4(mae)])
s.append(table(rows, [2.2 * cm, 1.4 * cm, 3.6 * cm, 2.2 * cm, 3 * cm, 3 * cm], small=True,
               shade_rows={r: "#F7F9FB" for r in range(7, 13)}))
s.append(Spacer(1, 4))
s.extend(bullets([
    "The constant (~0.27) is close to <b>mid-range</b> images, so the baseline already looks "
    "good there (MAE ~0.07-0.09). A model can only show real value in the low and high ranges.",
    "<b>Low range</b> (clean panels, largest group): the baseline over-predicts by ~0.21-0.23. "
    "<b>High range</b>: it under-predicts by ~0.38-0.44. A CNN that truly recognises clean vs "
    "heavily soiled panels must reduce these two numbers most.",
    "A model that only learns to output values near the mean could beat the overall MAE slightly "
    "while still failing at the extremes; the per-range table makes that visible.",
]))

# 7 loading speed
s.append(Paragraph("7. Loading speed (3.6)", H1))
s.append(table([
    ["Setting", "Measured (recorded run of data.py)", "Meaning"],
    ["num_workers = 0", "~301 images/s", "~1.8 min of loading per grouped-train epoch "
                                        "(32,158 images). Acceptable for Step 4."],
    ["num_workers = 4", "~32 images/s", "<b>Not a real throughput.</b> Only 11 batches were "
     "timed, and on Windows each worker is a new process (spawn) that re-imports torch and "
     "re-reads the split CSV. That start-up cost dominates such a short timing."],
], [3.2 * cm, 4.8 * cm, 9 * cm]))
s.append(Spacer(1, 4))
s.extend(bullets([
    "Decision: compare " + C % "--workers 0" + " vs " + C % "4" + " over a <b>full epoch</b> in "
    "Step 4 and keep the faster one. With " + C % "persistent_workers=True" + " the start-up "
    "is paid once, so 4 workers are expected to win over a full run, but this has not been "
    "measured yet and is not claimed.",
    "Small images (192x192 JPEG) make decoding cheap; the GPU forward/backward pass is likely "
    "to be the slower part.",
]))

# 8 outputs
s.append(Paragraph("8. Outputs produced", H1))
s.append(table([
    ["File", "Content"],
    ["data.py", "Shared config, transforms, SoilingDataset, make_loader; running it performs the "
                "sanity checks and the timing."],
    ["sanity_batch.png", "Figure 2 (train batch with jitter, grouped-test images)."],
    ["baseline.py", "Mean-predictor baseline (overall and per L range, printed)."],
    ["baseline_results.csv", "strategy, set, n, MAE, RMSE, R2 for val and test of both strategies."],
    [".venv/", "Python 3.12 environment with torch (cu128), torchvision, pandas, matplotlib, Pillow."],
    ["docs/step3_figs/jitter_extremes.png", "Figure 1 (made by this documentation script)."],
], [5.6 * cm, 11.4 * cm]))

# 9 decisions
s.append(Paragraph("9. Decisions and caveats", H1))
s.extend(bullets([
    "<b>One pipeline for everyone</b>: all training, evaluation, Grad-CAM and robustness code "
    "imports " + C % "data.py" + "; only " + C % "DATA_ROOT" + " differs per person.",
    "<b>Target unit fixed</b>: L as a fraction 0-1 (float32) everywhere in code; multiply by 100 "
    "only in tables and slides.",
    "<b>Preprocessing frozen</b>: RGB, resize 224, ToTensor, ImageNet normalisation; train-only "
    "brightness/contrast jitter 0.15. No flips, no crops, no augmentation on val/test.",
    "<b>Baseline = train mean of the same strategy</b>; the CNN is compared to the baseline of "
    "the same split and set, with MAE, RMSE, R<super>2</super> and per-range MAE.",
    "<b>Environment caveat</b>: system Python 3.15 cannot run PyTorch; the RTX 5060 requires the "
    "cu128 build. Always activate " + C % ".venv" + " first.",
    "<b>Timing caveat</b>: the 4-worker figure is a start-up artifact (Section 7). Also, the "
    "comment above the timing block in " + C % "data.py" + " says '2 workers, 320 images', "
    "while the code times 0 and 4 workers over 352 images; the printed numbers follow the code.",
    "<b>Sanity batch is random</b>: " + C % "python data.py" + " does not fix a seed, so the "
    "training images in " + C % "sanity_batch.png" + " differ between runs. The checks do not "
    "depend on which images are drawn.",
    "<b>Label noise</b>: visually similar soiling can have very different L (Section 5). This "
    "limits achievable accuracy and will be discussed with the results.",
]))

# 10 next
s.append(Paragraph("10. Next step (Step 4): ResNet-18 training", H1))
s.extend(bullets([
    "Pretrained ResNet-18; the final layer replaced by one linear output (predicted L, clamped "
    "to [0, 1] at evaluation). MSE loss.",
    "Phase A: train the new head only (2 epochs, AdamW, lr 1e-3). Phase B: fine-tune the whole "
    "network (up to 12 epochs, AdamW, lr 1e-4, weight decay 1e-4). Batch 64, bf16 autocast, "
    "seed 42.",
    "Early stopping on the validation MSE <b>of the same strategy</b> (patience 3); keep the best "
    "checkpoint; save per-image predictions for val and test.",
    "Run a short debug run first, then the grouped and random trainings; compare MAE, RMSE, "
    "R<super>2</super> and per-range MAE with the baseline tables in Section 6.",
]))
s.append(KeepTogether([Paragraph("10.1 Timeline", H2), table([
    ["Date", "Work"],
    ["19-20 Nov 2026", "Step 4 ResNet-18 training and evaluation; Grad-CAM; robustness test"],
    ["20 Nov 2026", "Code freeze, report; viva preparation in the evening"],
    ["21 Nov 2026", "Presentation"],
], [4 * cm, 13 * cm])]))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Step 3: Data Pipeline, Environment and Baseline",
                        author="IDSE Solar Soiling Team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("sizes ok:", sizes_ok, "| batch ok:", batch_ok, "| baseline matches csv:", base_ok)
print("batch:", batch, "| raw:", raw_info)
print("per-range:", [(a, b, c, d, round(f, 4)) for a, b, c, d, _, f in per_range])
print("Wrote", PDF)
