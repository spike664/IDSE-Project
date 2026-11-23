"""Build docs/Step2_Leakage_Aware_Splits.pdf (Step 2 documentation).

Usage (from the project folder):
    python -m pip install reportlab pandas matplotlib
    python docs/build_step2_pdf.py

All numbers are recomputed from the frozen split file splits/splits.csv.
The day-assignment search of step2_splits.py is re-run here only to confirm that
it reproduces the frozen grouped split and to report how many candidates it checked;
the split files themselves are not written or changed.
"""
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
SPL = ROOT / "splits"
FIG_DIR = ROOT / "docs" / "step2_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF = ROOT / "docs" / "Step2_Leakage_Aware_Splits.pdf"
SETS = ("train", "val", "test")
SET_COLOR = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}  # same as split_L_hist.png

# ---------------------------------------------------------------- numbers
df = pd.read_csv(SPL / "splits.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
N = len(df)
ts = (df["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds()

day = df.groupby("day").agg(
    n=("L", "size"), L_mean=("L", "mean"), L_median=("L", "median"),
    L_min=("L", "min"), L_max=("L", "max"), I_mean=("I", "mean"),
    first=("timestamp", "min"), last=("timestamp", "max"),
    sessions=("group_id", "nunique"), split=("grouped_split", "first"))
days = list(day.index)

grp = df.groupby("group_id").agg(n=("L", "size"), day=("day", "first"),
                                 n_days=("day", "nunique"))
sess_per_day = grp.groupby("day").size()

# split sizes / L stats
def split_stats(col):
    rows = []
    for s in SETS:
        sub = df[df[col] == s]
        rows.append((s, len(sub), 100 * len(sub) / N, sub["L"].mean(),
                     sub["L"].median(), sub["L"].std(), sub["L"].min(), sub["L"].max()))
    return rows

# share of images per L band
BANDS = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
BAND_NAMES = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.8", "0.8-1.0"]

def band_shares(col):
    out = []
    for s in SETS:
        L = df.loc[df[col] == s, "L"]
        out.append([100 * ((L >= a) & (L < b)).mean() for a, b in BANDS])
    return out

# 20-bin total-variation distance to train (same measure the search used)
BINS = np.linspace(0, 1.0001, 21)

def tv(a, b):
    ha, hb = np.histogram(a, bins=BINS)[0], np.histogram(b, bins=BINS)[0]
    return 0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum()

def tv_to_train(col):
    tr = df.loc[df[col] == "train", "L"]
    return {s: tv(df.loc[df[col] == s, "L"], tr) for s in ("val", "test")}

# leakage evidence: time to nearest train image
def nearest_train_gap(col):
    tr = np.sort(ts[df[col] == "train"].values)
    out = {}
    for s in ("val", "test"):
        t = ts[df[col] == s].values
        pos = np.searchsorted(tr, t)
        left = np.abs(t - tr[np.clip(pos - 1, 0, len(tr) - 1)])
        right = np.abs(tr[np.clip(pos, 0, len(tr) - 1)] - t)
        out[s] = np.minimum(left, right)
    return out

gaps = {c: nearest_train_gap(c) for c in ("random_split", "grouped_split")}

# re-run the day-assignment search (read-only) to confirm reproducibility
cnt = day["n"].values.astype(float)
hist = np.array([np.histogram(df.loc[df["day"] == d, "L"], bins=BINS)[0] for d in days],
                dtype=float)
idx = range(len(days))

def tvh(a, b):
    return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()

n_cand, best, best_cost, costs = 0, None, 1e9, []
for k_te in (2, 3, 4):
    for te in itertools.combinations(idx, k_te):
        f_te = cnt[list(te)].sum() / N
        if not 0.12 <= f_te <= 0.18:
            continue
        rest = [i for i in idx if i not in te]
        for k_va in (1, 2, 3):
            for va in itertools.combinations(rest, k_va):
                f_va = cnt[list(va)].sum() / N
                if not 0.12 <= f_va <= 0.18:
                    continue
                tr = [i for i in rest if i not in va]
                h_tr = hist[tr].sum(0)
                cost = (tvh(hist[list(te)].sum(0), h_tr) + tvh(hist[list(va)].sum(0), h_tr)
                        + 2 * (abs(f_te - 0.15) + abs(f_va - 0.15)))
                n_cand += 1
                costs.append(cost)
                if cost < best_cost:
                    best, best_cost = (te, va), cost
search_te = sorted(days[i] for i in best[0])
search_va = sorted(days[i] for i in best[1])
frozen_te = sorted(day.index[day["split"] == "test"])
frozen_va = sorted(day.index[day["split"] == "val"])
search_matches = (search_te == frozen_te) and (search_va == frozen_va)
costs = np.array(costs)

# integrity checks on the frozen files
checks = {
    "filenames unique": df["filename"].is_unique,
    "every day in exactly one grouped set": bool((df.groupby("day")["grouped_split"].nunique() == 1).all()),
}
for name in ("random", "grouped"):
    parts = [pd.read_csv(SPL / f"{name}_{s}.csv") for s in SETS]
    union = pd.concat(parts)["filename"]
    checks[f"{name}_*.csv: sets disjoint and cover all {N:,} images"] = (
        union.is_unique and len(union) == N and set(union) == set(df["filename"]))
    checks[f"{name}_*.csv sizes match splits.csv"] = all(
        len(p) == (df[f"{name}_split"] == s).sum() for p, s in zip(parts, SETS))
sess_cross_days = int((grp["n_days"] > 1).sum())

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(12, 4.2))
data = [df.loc[df["day"] == d, "L"].values for d in days]
bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6)
for patch, d in zip(bp["boxes"], days):
    patch.set_facecolor(SET_COLOR[day.loc[d, "split"]]); patch.set_alpha(0.55)
for med in bp["medians"]:
    med.set_color("black")
ax.plot(range(1, len(days) + 1), day["L_mean"].values, "kD", ms=4, label="mean")
ax.set_xticks(range(1, len(days) + 1))
ax.set_xticklabels([f"{d[8:]} Jun\nn={day.loc[d, 'n']:,}" for d in days], fontsize=8)
ax.set_ylabel("L (fraction)"); ax.set_ylim(0, 1)
ax.set_title("L per day (box = 25-75%, whiskers = 5-95%), coloured by grouped-split set")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=SET_COLOR[s], alpha=0.55, label=s) for s in SETS]
          + [plt.Line2D([], [], color="k", marker="D", ls="", ms=4, label="mean")],
          loc="upper left", fontsize=8, ncol=4)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
FIG_DAY = FIG_DIR / "L_per_day_by_set.png"
plt.savefig(FIG_DAY, dpi=130); plt.close()

# ---------------------------------------------------------------- styles (same as Step 1)
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


def p(t):
    return Paragraph(t, B)


def bullets(items):
    return [Paragraph(t, BL, bulletText="\u2022") for t in items]


def table(rows, widths, header=True, small=False, shade_rows=None):
    st = CELLS if small else CELL
    data = [[Paragraph(str(c), st) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
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


def f3(x):
    return f"{x:.3f}"


def hms(sec):
    sec = float(sec)
    if sec < 120:
        return f"{sec:,.0f} s"
    if sec < 7200:
        return f"{sec:,.0f} s (~{sec / 60:.0f} min)"
    return f"{sec:,.0f} s (~{sec / 3600:.1f} h)"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "IDSE project - Solar panel soiling power-loss regression - Step 2 documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def daylist(s):
    return ", ".join(f"{d[8:]}" for d in day.index[day["split"] == s]) + " Jun"


# ---------------------------------------------------------------- content
s = []
s.append(Paragraph("Step 2: Leakage-Aware Train / Validation / Test Splits", ss["Title"]))
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
                   "Status: splits are <b>frozen</b>.", CAP))

# 1
s.append(Paragraph("1. Purpose of Step 2 and where it fits", H1))
s.append(p(
    "Step 1 showed that the camera takes a picture of the same panel about every 5 seconds, so "
    "neighbouring frames are almost identical (same dust, nearly the same light). Step 2 decides "
    "<b>which images the model may learn from and which images it is judged on</b>. If this is "
    "done carelessly, test images have near-copies in the training set, and the test score "
    "measures memorisation of a moment in time instead of the ability to read soiling from a "
    "new image."))
s.append(p(
    "We therefore build <b>two</b> split strategies and keep both: a <b>random</b> 70/15/15 "
    "split (the naive approach most tutorials use) and a <b>grouped-by-day</b> 70/15/15 split "
    "(the leakage-aware approach). The model in Step 4 is trained and evaluated on both, and the "
    "difference between the two scores is one of the main findings of the project. Step 2 also "
    "measures, before any training, how close the val/test images are in time to the training "
    "images under each strategy; this is the direct evidence for (or against) leakage."))
s.append(p("Every team member uses the same split files. Nobody creates their own split, and the "
           "split script is not re-run with changed settings."))

# 2
s.append(Paragraph("2. What we did and why", H1))
s.append(table([
    ["#", "Action", "Why"],
    ["2.1", "Summarised L for each of the 15 days (n, mean, median, min, max, mean I, first and "
            "last capture).",
     "To see whether soiling drifts steadily over time or jumps between days. This decides "
     "whether a simple chronological split is acceptable and which grouping unit to use."],
    ["2.2", "Cut the time series into <b>sessions</b>: a new session starts when the gap to the "
            "previous image exceeds 600 s (10 min). Stored as <font face='Courier'>group_id</font>.",
     "Sessions are the natural runs of continuous capture. They were the candidate alternative to "
     "days as the grouping unit."],
    ["2.3", "Random split: shuffle all images with seed 42, take 70% / 15% / 15%.",
     "The usual baseline split. Needed as the comparison point that shows how optimistic a "
     "naive split is."],
    ["2.4", "Grouped split: assign each <b>whole day</b> to exactly one set. Exhaustively search "
            "all day-to-set assignments that give ~70/15/15 images and choose the one whose L "
            "distribution in val and test is closest to train.",
     "Removes near-duplicate frames across sets. Searching (instead of picking days by hand) "
     "keeps the label distributions comparable, so a score gap is caused by leakage and not by "
     "an easier/harder test set chosen by accident."],
    ["2.5", "Verified sizes, L statistics per set, that no day is in two grouped sets, and that "
            "filenames are unique.",
     "Catches assignment bugs before anyone trains on the files."],
    ["2.6", "For every val/test image measured the time to its nearest training image, for both "
            "strategies.",
     "Quantifies leakage directly: if most test images have a training image a few seconds "
     "away, the random split is leaking."],
    ["2.7", "Saved one combined file and six per-set CSV files (no absolute paths) and a "
            "histogram plot.",
     "One shared, portable set of files for all teammates; each person points "
     "<font face='Courier'>DATA_ROOT</font> to their own image folder."],
], [0.9 * cm, 7.6 * cm, 8.5 * cm]))
s.append(Paragraph("2.8 Scripts and commands", H2))
s.append(p("Scripts (project folder): <font face='Courier'>step2a_summary.py</font> (2.1, 2.2, "
           "print-only) and <font face='Courier'>step2_splits.py</font> (2.3-2.7, writes "
           "<font face='Courier'>splits/</font>). Both read only "
           "<font face='Courier'>step1_outputs/dataset_manifest.csv</font>; raw images are never "
           "opened or changed."))
s.append(Preformatted(
    "cd C:\\Users\\manie\\OneDrive\\Desktop\\solar-soiling\n"
    "python step2a_summary.py      # per-day table and session sizes\n"
    "python step2_splits.py        # builds, verifies and saves the splits (already run, FROZEN)",
    CODE))
s.append(table([
    ["Setting", "Value", "Reason"],
    ["Random seed", "42", "Reproducible shuffle (project-wide seed)."],
    ["Split ratio", "70 / 15 / 15", "Enough training data; val and test large enough for stable "
                                    "metrics (~6,800 images each)."],
    ["Session gap", "600 s", "Step 1: only 56 gaps exceed 10 min, so they mark real breaks, not "
                             "normal 5 s capture jitter."],
    ["Histogram bins for matching", "20 equal bins over [0, 1]", "Fine enough to see the bumps in "
                                                                 "L, coarse enough to be stable per day."],
    ["Allowed val/test size", "12-18% of images each", "Days are indivisible, so exactly 15% is "
                                                       "not possible."],
], [4 * cm, 4.2 * cm, 8.8 * cm]))

# 3
s.append(Paragraph("3. Results", H1))
s.append(Paragraph("3.1 Per-day label summary (2.1)", H2))
rows = [["Date", "Images", "Sessions", "L mean", "L median", "L min", "L max", "I mean", "First - last"]]
for d, r in day.iterrows():
    rows.append([d[5:], f"{r['n']:,}", str(r["sessions"]), f3(r["L_mean"]), f3(r["L_median"]),
                 f3(r["L_min"]), f3(r["L_max"]), f3(r["I_mean"]),
                 f"{r['first']:%H:%M} - {r['last']:%H:%M}"])
s.append(table(rows, [1.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm,
                      1.6 * cm, 4.3 * cm], small=True))
s.append(Spacer(1, 4))
s.extend(bullets([
    "<b>Soiling is not monotonic over time.</b> Mean L rises from 0.120 (13 Jun) to 0.507 "
    "(16 Jun), falls to ~0.19 (20-21 Jun), rises to 0.571 (23 Jun), drops to 0.023 (25 Jun) and "
    "rises again to 0.678 (30 Jun). The drops are consistent with cleaning or rain resetting the "
    "panel.",
    "<b>Consequence 1: no chronological split.</b> Taking the last ~15% of time as test would put "
    "only 28-30 Jun (mean L 0.52-0.68) in test. The model would be trained mostly on cleaner days "
    "and tested only on heavily soiled ones, which mixes leakage control with a large label shift "
    "and makes the result hard to interpret.",
    "<b>Consequence 2: days differ strongly in L</b>, so which days go into val/test changes the "
    "label distribution a lot. That is why the day assignment is chosen by a search that matches "
    "distributions (Section 3.3) instead of by hand.",
    "Within a day L still varies widely (see L min/max), so each day contributes a range of "
    "soiling levels, not a single value.",
]))
s.append(KeepTogether([
    Image(str(FIG_DAY), width=17 * cm, height=17 * cm * 4.2 / 12),
    Paragraph("Figure 1. Distribution of L on each day, coloured by the grouped-split set the day "
              "was assigned to (made by docs/build_step2_pdf.py from splits/splits.csv).", CAP),
]))

s.append(Paragraph("3.2 Sessions (2.2) and choice of grouping unit", H2))
s.append(table([
    ["Quantity", "Value"],
    ["Number of sessions (gap > 600 s)", f"{len(grp)}"],
    ["Session size (images)", f"min {grp['n'].min():,} / median {grp['n'].median():,.0f} / "
                              f"max {grp['n'].max():,}"],
    ["Sessions per day", f"min {sess_per_day.min()} / median {sess_per_day.median():.0f} / "
                         f"max {sess_per_day.max()}"],
    ["Sessions spanning two days", f"{sess_cross_days}"],
], [7 * cm, 10 * cm]))
s.append(Spacer(1, 4))
s.append(p("<b>Decision: the split unit is the day, not the session.</b> Reasons:"))
s.extend(bullets([
    "Sessions on the same day are separated only by a pause in capture (e.g. a few hours); the "
    "dust on the panel is the same. A session-level split could put the morning of a day in train "
    "and its afternoon in test, which is still leakage of the same soiling state, just with a "
    "larger time gap.",
    "Session sizes are very uneven (1 to 5,093 images), so they are not a cleaner unit for "
    "balancing 70/15/15 anyway.",
    "<font face='Courier'>group_id</font> is still saved in the split files as metadata, in case "
    "it is useful for analysis later.",
]))

s.append(Paragraph("3.3 How the grouped split was chosen (2.4)", H2))
s.append(p(
    "Each candidate assignment puts 2-4 days in test and 1-3 days in val, with each of val and "
    "test holding 12-18% of all images; the remaining days are train. For each candidate we "
    "computed the 20-bin histogram of L in train, val and test and scored it with"))
s.append(Preformatted(
    "cost = TV(test, train) + TV(val, train) + 2 * (|f_test - 0.15| + |f_val - 0.15|)\n"
    "TV(a, b) = 0.5 * sum_bins | a_bin/sum(a) - b_bin/sum(b) |   (total-variation distance, 0..1)\n"
    "f_set    = fraction of all images in that set", CODE))
s.append(p(
    f"The search checked <b>{n_cand:,}</b> valid candidates (cost range {costs.min():.3f} to "
    f"{costs.max():.3f}, median {np.median(costs):.3f}) and kept the one with the lowest cost "
    f"({best_cost:.3f}). The size penalty is weighted 2x so that a slightly better distribution "
    "match cannot buy a badly unbalanced split. Re-running the search while building this "
    f"document {'<b>reproduces the frozen assignment exactly</b>' if search_matches else '<b>did NOT reproduce the frozen assignment - investigate</b>'}."))
s.append(p(
    "Note: the search uses only the <b>label distribution</b> of each day, never a model or a "
    "model score, so it does not tune the split to make our model look good."))
s.append(table([
    ["Set", "Days (June 2017)", "Images", "Share"],
    ["train", daylist("train"), f"{(df['grouped_split'] == 'train').sum():,}",
     f"{100 * (df['grouped_split'] == 'train').mean():.1f}%"],
    ["val", daylist("val"), f"{(df['grouped_split'] == 'val').sum():,}",
     f"{100 * (df['grouped_split'] == 'val').mean():.1f}%"],
    ["test", daylist("test"), f"{(df['grouped_split'] == 'test').sum():,}",
     f"{100 * (df['grouped_split'] == 'test').mean():.1f}%"],
], [2 * cm, 9 * cm, 3 * cm, 3 * cm]))

s.append(Paragraph("3.4 Split sizes and label statistics (2.3-2.5)", H2))
rows = [["Strategy", "Set", "Images", "Share", "L mean", "L median", "L std", "L min", "L max"]]
for col, nm in (("random_split", "random"), ("grouped_split", "grouped")):
    for st, n_, pct, mn, md, sd, lo, hi in split_stats(col):
        rows.append([nm, st, f"{n_:,}", f"{pct:.1f}%", f3(mn), f3(md), f3(sd), f3(lo), f3(hi)])
s.append(table(rows, [2 * cm, 1.4 * cm, 1.9 * cm, 1.6 * cm, 1.8 * cm, 1.9 * cm, 1.7 * cm,
                      1.7 * cm, 1.7 * cm], small=True, shade_rows={4: "#F7F9FB", 5: "#F7F9FB",
                                                                     6: "#F7F9FB"}))
s.append(Spacer(1, 6))
rows = [["Strategy", "Set"] + [f"L {b}" for b in BAND_NAMES] + ["TV to train"]]
for col, nm in (("random_split", "random"), ("grouped_split", "grouped")):
    tvs = tv_to_train(col)
    for st, sh in zip(SETS, band_shares(col)):
        rows.append([nm, st] + [f"{x:.1f}%" for x in sh]
                    + ["-" if st == "train" else f"{tvs[st]:.3f}"])
s.append(KeepTogether([
    Paragraph("Share of images per L band, and distance of each L distribution to train", H2),
    table(rows, [2 * cm, 1.4 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 3.1 * cm],
          small=True, shade_rows={4: "#F7F9FB", 5: "#F7F9FB", 6: "#F7F9FB"})]))
s.append(Spacer(1, 6))
s.append(KeepTogether([
    Image(str(SPL / "split_L_hist.png"), width=17 * cm, height=17 * cm * 480 / 1440),
    Paragraph("Figure 2. L distribution of train / val / test (density, 40 bins). Left: random "
              "split. Right: grouped-by-day split. (splits/split_L_hist.png)", CAP),
]))
s.extend(bullets([
    "<b>Random split</b>: the three sets have practically identical distributions "
    "(means 0.266 / 0.263 / 0.262). This is expected from shuffling 45k images.",
    "<b>Grouped split</b>: sizes are close to 70/15/15 (70.3 / 15.0 / 14.7%), but the "
    "distributions differ, because whole days move together. Val has more near-clean images "
    "(median 0.094); test has fewer near-zero images, clear bumps around 0.3 and 0.65, and "
    "<b>almost no images above L = 0.8</b> (see the band table). Very heavy soiling is therefore "
    "evaluated mainly on val, and even there it is a small share.",
    "This shift is intended and realistic: a deployed model sees new days whose soiling mix is "
    "not the same as training. Grouped test scores are expected to be worse than random-split "
    "scores; part of that gap may come from this shift, not only from removed leakage, and the "
    "report must say so.",
    "Training covers the full L range (0 to ~1), so no test value lies outside what the model "
    "has seen.",
]))

s.append(Paragraph("3.5 Leakage evidence (2.6)", H2))
s.append(p("For each val/test image: time difference to the closest training image (in seconds)."))
rows = [["Strategy", "Set", "Median", "90th percentile", "Max", "Within 10 s"]]
for col, nm in (("random_split", "random"), ("grouped_split", "grouped")):
    for st in ("val", "test"):
        a = gaps[col][st]
        rows.append([nm, st, hms(np.median(a)), hms(np.percentile(a, 90)), hms(a.max()),
                     f"{(a <= 10).mean() * 100:.1f}%"])
s.append(table(rows, [2 * cm, 1.3 * cm, 3.5 * cm, 3.8 * cm, 3.8 * cm, 2.6 * cm]))
s.append(Spacer(1, 4))
s.extend(bullets([
    "<b>Random split</b>: the typical val/test image has a training image only <b>5 seconds</b> "
    "away, and more than 92% have one within 10 s. Such a frame shows the same dust under almost "
    "the same light, so the model can score well by recognising the moment rather than by "
    "judging soiling.",
    "<b>Grouped split</b>: the nearest training image is typically <b>17-18 hours</b> away "
    "(a different day) and no val/test image has a training image within 10 s.",
    "This is measured before training, so it is evidence that leakage is <b>possible</b> under "
    "the random split. Whether it actually <b>inflates the scores</b> is tested in Step 4 by "
    "comparing test metrics of both strategies. We do not claim inflated scores until those "
    "numbers exist.",
]))

s.append(Paragraph("3.6 Integrity checks on the saved files", H2))
rows = [["Check", "Result"]] + [[k, "OK" if v else "FAILED"] for k, v in checks.items()]
s.append(table(rows, [12 * cm, 5 * cm]))

# 4
s.append(Paragraph("4. Outputs produced", H1))
s.append(table([
    ["File", "Content"],
    ["splits/splits.csv", "All images, one row each. Columns: filename, timestamp, L, I, day, "
                          "group_id, random_split, grouped_split."],
    ["splits/random_{train,val,test}.csv", "Random split, one file per set. Columns: filename, "
                                           "timestamp, L, I, day, group_id."],
    ["splits/grouped_{train,val,test}.csv", "Grouped-by-day split, one file per set. Same "
                                            "columns."],
    ["splits/split_L_hist.png", "Figure 2."],
    ["step2a_summary.py", "Per-day table and session sizes (prints only)."],
    ["step2_splits.py", "Builds, verifies and saves the splits."],
    ["docs/step2_figs/L_per_day_by_set.png", "Figure 1 (made by the documentation script)."],
], [6.5 * cm, 10.5 * cm]))
s.append(p("The CSVs hold <b>filenames only, no absolute paths</b>. Loaders build the path as "
           "<font face='Courier'>DATA_ROOT / 'PanelImages' / filename</font>, where "
           "<font face='Courier'>DATA_ROOT</font> is set per teammate."))

# 5
s.append(Paragraph("5. Decisions and caveats", H1))
s.extend(bullets([
    "<b>Two strategies, both kept</b>: random (naive reference) and grouped-by-day "
    "(leakage-aware, the main result).",
    "<b>Split unit = day</b>: same-day images share the same dust state; sessions would leak it.",
    "<b>No chronological split</b>: soiling resets between days, so the last days are not "
    "representative.",
    "<b>Day assignment by search</b> on L distributions only (no model involved), size within "
    "12-18% for val and test.",
    "<b>Frozen</b>: the split files are final. step2_splits.py is not re-run with other settings; "
    "nobody makes a private split.",
    "<b>Bug found and fixed</b>: the first run of the leakage block printed times about 1000x too "
    "small, because newer pandas stores these timestamps at a coarser resolution than nanoseconds, "
    "so converting the raw integer values to seconds was wrong. Fixed by computing seconds as "
    "<font face='Courier'>(timestamp - 1970-01-01).dt.total_seconds()</font>. Split assignments and "
    "L statistics did not depend on this and were unchanged; all leakage numbers here are from "
    "the corrected computation.",
    "<b>Limitation</b>: grouped test is only 2 days and contains little very-high-L data; val has "
    "a different mix again. With 15 days from one site and one camera, results describe "
    "generalisation to new days at this site, not to other panels or locations.",
    "<b>Validation choice</b>: early stopping and model selection must use the val set of the "
    "<b>same</b> strategy. Using a random val set while reporting grouped test would reintroduce "
    "leakage into model selection.",
]))

# 6
s.append(Paragraph("6. Next step (Step 3): data pipeline and baseline", H1))
s.extend(bullets([
    "Check the PyTorch/torchvision install (local GPU or Colab).",
    "Dataset/DataLoader that reads the split CSVs and <font face='Courier'>DATA_ROOT</font>; "
    "fixed preprocessing: RGB, resize 192 to 224, tensor, ImageNet normalisation "
    "(mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).",
    "Training-only augmentation: small brightness/contrast jitter. No heavy crops, no horizontal "
    "flip. No augmentation on val/test.",
    "Sanity batch and an augmentation grid image; loading-speed check.",
    "Mean-predictor baseline (predict the training-set mean of L) with MAE, RMSE and "
    "R<super>2</super> on val and test, for <b>both</b> split strategies. This is the bar the "
    "ResNet-18 must beat.",
]))
s.append(KeepTogether([Paragraph("6.1 Timeline", H2), table([
    ["Date", "Work"],
    ["19-20 Nov 2026", "Step 3 pipeline and baseline; Step 4 ResNet-18 training and evaluation"],
    ["20 Nov 2026", "Code freeze, report; viva preparation in the evening"],
    ["21 Nov 2026", "Presentation"],
], [4 * cm, 13 * cm])]))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Step 2: Leakage-Aware Train / Validation / Test Splits",
                        author="IDSE Solar Soiling Team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("search reproduces frozen split:", search_matches, "| candidates:", n_cand)
print("checks:", checks)
print("Wrote", PDF)
