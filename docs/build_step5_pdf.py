"""Build docs/Step5_Evaluation_and_Error_Analysis.pdf (Step 5 documentation).

Usage (from the project folder, venv active):
    python docs/build_step5_pdf.py

Nothing is retrained or re-predicted here. All numbers come from the files written by
evaluate.py (outputs/eval/*.csv) and from the per-image prediction files of Step 4
(outputs/preds/*.csv) joined with splits/splits.csv. Every table taken from outputs/eval is
recomputed from the prediction files and cross-checked. The extra diagnostics in this document
(per-day timelines, error share per day, distinct label readings in the worst cases,
low-irradiance check) are also computed only from those saved files and are labelled as such.
Raw images are opened read-only to make the thumbnail figure.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
EV = OUT / "eval"
FIG_DIR = ROOT / "docs" / "step5_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF = ROOT / "docs" / "Step5_Evaluation_and_Error_Analysis.pdf"
DATA_ROOT = Path(os.environ.get(
    "SOLAR_DATA_ROOT",
    r"C:\Users\manie\Downloads\SolarPanelSoilingImageDataset\Solar_Panel_Soiling_Image_dataset\PanelImages"))
STRATS = ("random", "grouped")
SETS = ("val", "test")
BINS = [(0.0, 0.15, "low"), (0.15, 0.5, "mid"), (0.5, 1.01, "high")]


def metrics(y, p):
    e = y - p
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}


# ---------------------------------------------------------------- load
sp = pd.read_csv(ROOT / "splits" / "splits.csv", parse_dates=["timestamp"])
runs, train_mean = {}, {}
for st in STRATS:
    train_mean[st] = sp.loc[sp[f"{st}_split"] == "train", "L"].mean()
    for sb in SETS:
        d = pd.read_csv(OUT / "preds" / f"resnet18_{st}_{sb}.csv").merge(
            sp[["filename", "timestamp", "day", "I", "group_id"]], on="filename", how="left")
        d["err"] = d["L_pred"] - d["L_true"]
        runs[(st, sb)] = d

ev_all = pd.read_csv(EV / "comparison_all.csv")
ev_rng = pd.read_csv(EV / "error_by_range.csv")
ev_day = {(st, sb): pd.read_csv(EV / f"per_day_error_{st}_{sb}.csv") for st in STRATS for sb in SETS}
w20 = {st: pd.read_csv(EV / f"worst20_{st}_test.csv") for st in STRATS}

# ---------------------------------------------------------------- recompute + cross-check
comp, checks = {}, []
for (st, sb), d in runs.items():
    y, p = d.L_true.to_numpy(float), d.L_pred.to_numpy(float)
    b = metrics(y, np.full_like(y, train_mean[st]))
    c = metrics(y, p)
    comp[(st, sb)] = {"n": len(d), "b": b, "c": c, "red": 100 * (1 - c["MAE"] / b["MAE"]),
                      "bias": d.err.mean(), "merge_ok": d.day.notna().all()}
    r = ev_all[(ev_all.split == st) & (ev_all.set == sb)].iloc[0]
    checks.append(np.allclose([b["MAE"], b["RMSE"], b["R2"], c["MAE"], c["RMSE"], c["R2"]],
                              r[["base_MAE", "base_RMSE", "base_R2", "CNN_MAE", "CNN_RMSE", "CNN_R2"]]
                              .to_numpy(float), atol=1e-4) and r.n == len(d))
rng = {}
for st in STRATS:
    for sb in SETS:
        d = runs[(st, sb)]
        for lo, hi, k in BINS:
            mk = (d.L_true >= lo) & (d.L_true < hi)
            rng[(st, sb, k)] = {"n": int(mk.sum()), "MAE": d.err[mk].abs().mean(),
                                "bias": d.err[mk].mean(),
                                "base": (d.L_true[mk] - train_mean[st]).abs().mean()}
    for _, row in ev_rng[ev_rng.split == st].iterrows():
        v = rng[(st, "test", row["range"])]
        checks.append(abs(v["MAE"] - row.MAE) < 1e-4 and abs(v["bias"] - row.bias) < 1e-4
                      and v["n"] == row.n)
for (st, sb), d in runs.items():
    g = d.groupby("day").agg(n=("err", "size"), MAE=("err", lambda e: e.abs().mean()),
                             bias=("err", "mean"), Lt=("L_true", "mean"), Lp=("L_pred", "mean"))
    e = ev_day[(st, sb)].set_index("day")
    checks.append(np.allclose(g[["MAE", "bias", "Lt", "Lp"]].to_numpy(),
                              e[["MAE", "bias", "L_true_mean", "L_pred_mean"]].to_numpy(), atol=1e-4))
for st in STRATS:
    d = runs[(st, "test")]
    top = d.assign(a=d.err.abs()).sort_values("a", ascending=False).head(20)
    checks.append(list(top.filename) == list(w20[st].filename))
all_ok = all(checks)
merge_ok = all(v["merge_ok"] for v in comp.values())

# ---------------------------------------------------------------- extra diagnostics (from saved files)
gtest, gval = runs[("grouped", "test")], runs[("grouped", "val")]
# share of total absolute error per day
share = {}
for nm, d in (("val", gval), ("test", gtest)):
    tot = d.err.abs().sum()
    share[nm] = (d.groupby("day").err.apply(lambda e: e.abs().sum()) / tot * 100)
# grouped test without 21 Jun / without 29 Jun
excl = {}
for day in ("2017-06-21", "2017-06-29"):
    k = gtest[gtest.day != day]
    excl[day] = metrics(k.L_true.to_numpy(), k.L_pred.to_numpy())
# 21 Jun per L range
d21 = gtest[gtest.day == "2017-06-21"]
r21 = {}
for lo, hi, k in BINS:
    mk = (d21.L_true >= lo) & (d21.L_true < hi)
    r21[k] = (int(mk.sum()), d21.err[mk].abs().mean(), d21.err[mk].mean())
# worst-20 structure
wg = w20["grouped"].merge(sp[["filename", "group_id", "timestamp"]], on="filename")
n_read = wg[["L_true", "I"]].drop_duplicates().shape[0]
n_sess = wg.group_id.nunique()
wg = wg.sort_values("timestamp").reset_index(drop=True)
wg["win"] = (wg.timestamp.diff().dt.total_seconds().fillna(1e9) > 300).cumsum()  # 5-min windows
n_win = wg.win.nunique()
wg["sess"] = wg.day.str[-2:] + " Jun " + wg.groupby("win").timestamp.transform("min").dt.strftime("%H:%M")
# big-error counts
n_big = int((gtest.err.abs() > 0.5).sum())
n_big_under = int((gtest.err < -0.5).sum())
n_true_hi_pred_lo = int(((gtest.L_true >= 0.8) & (gtest.L_pred < 0.2)).sum())
n_true_hi = int((gtest.L_true >= 0.8).sum())
# low-irradiance check (I < 0.1)
lowI = {}
for st in STRATS:
    for sb in SETS:
        d = runs[(st, sb)]
        k = d[d.I >= 0.1]
        lowI[(st, sb)] = {"n_low": int((d.I < 0.1).sum()), "frac": float((d.I < 0.1).mean()),
                          "MAE_lowI": d.err[d.I < 0.1].abs().mean(),
                          **metrics(k.L_true.to_numpy(), k.L_pred.to_numpy())}
# bf16 quantisation of predictions
n_unique_pred = gtest.L_pred.nunique()
frac0 = float((gtest.L_pred <= 0).mean())

# ---------------------------------------------------------------- figure A: per-day grouped
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
days = [("val", d) for d in ev_day[("grouped", "val")].day] + \
       [("test", d) for d in ev_day[("grouped", "test")].day]
lab = [f"{d[-2:]} Jun\n({s})" for s, d in days]
mae = [ev_day[("grouped", s)].set_index("day").loc[d, "MAE"] for s, d in days]
lt = [ev_day[("grouped", s)].set_index("day").loc[d, "L_true_mean"] for s, d in days]
lp = [ev_day[("grouped", s)].set_index("day").loc[d, "L_pred_mean"] for s, d in days]
x = np.arange(len(days))
cols = ["#7F9CC0" if s == "val" else "#1F77B4" for s, _ in days]
axes[0].bar(x, mae, color=cols)
for i, v in enumerate(mae):
    axes[0].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
axes[0].axhline(comp[("grouped", "test")]["c"]["MAE"], ls="--", c="gray",
                label=f"grouped test overall {comp[('grouped', 'test')]['c']['MAE']:.4f}")
axes[0].set_xticks(x); axes[0].set_xticklabels(lab); axes[0].set_ylabel("MAE (fraction)")
axes[0].set_title("Grouped split: MAE per day", fontsize=10); axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.3)
axes[1].bar(x - 0.2, lt, 0.4, label="mean true L", color="#B0B7C3")
axes[1].bar(x + 0.2, lp, 0.4, label="mean predicted L", color="#1F77B4")
axes[1].set_xticks(x); axes[1].set_xticklabels(lab); axes[1].set_ylabel("L (fraction)")
axes[1].set_title("Grouped split: mean true vs predicted L per day", fontsize=10)
axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
plt.tight_layout()
FIG_DAY = FIG_DIR / "per_day_grouped.png"
plt.savefig(FIG_DAY, dpi=120); plt.close()

# ---------------------------------------------------------------- figure B: timelines of grouped test days
fig, axes = plt.subplots(2, 1, figsize=(13, 6.6))
for a, day in zip(axes, ("2017-06-21", "2017-06-29")):
    d = gtest[gtest.day == day].sort_values("timestamp")
    hrs = d.timestamp.dt.hour + d.timestamp.dt.minute / 60 + d.timestamp.dt.second / 3600
    a.scatter(hrs, d.L_true, s=3, c="black", label="true L")
    a.scatter(hrs, d.L_pred, s=3, c="#D62728", alpha=0.6, label="predicted L")
    a2 = a.twinx()
    a2.plot(hrs, d.I, c="#2CA02C", lw=0.6, alpha=0.6)
    a2.set_ylabel("irradiance I", color="#2CA02C", fontsize=8); a2.set_ylim(0, 1.1)
    a.set_ylim(-0.03, 1.03); a.set_ylabel("L (fraction)")
    m = metrics(d.L_true.to_numpy(), d.L_pred.to_numpy())
    a.set_title(f"{day[-2:]} Jun 2017 (grouped test, n={len(d):,}, MAE {m['MAE']:.4f})", fontsize=10)
    a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left", markerscale=4)
axes[1].set_xlabel("time of day (h)")
plt.tight_layout()
FIG_TL = FIG_DIR / "timeline_grouped_test.png"
plt.savefig(FIG_TL, dpi=110); plt.close()

# ---------------------------------------------------------------- figure C: worst-case thumbnails
picks = wg.groupby("win").head(1)          # first frame of each short time window
ref = gtest[(gtest.err.abs() < 0.03) & (gtest.L_true.between(0.6, 0.7))].sample(1, random_state=0)
ref_clean = gtest[(gtest.err.abs() < 0.02) & (gtest.L_true < 0.05)].sample(1, random_state=0)
tiles = [(r.filename, f"WORST: {r.sess}\ntrue {r.L_true:.3f}  pred {r.L_pred:.3f}\nI {r.I:.3f}")
         for r in picks.itertuples()]
tiles += [(r.filename, f"REFERENCE (correct)\n{r.day[-2:]} Jun {r.timestamp:%H:%M}\n"
                       f"true {r.L_true:.3f}  pred {r.L_pred:.3f}")
          for r in pd.concat([ref, ref_clean]).itertuples()]
img_ok = all((DATA_ROOT / f).exists() for f, _ in tiles)
FIG_TH = FIG_DIR / "worst_thumbnails.png"
if img_ok:
    fig, axes = plt.subplots(2, 4, figsize=(13, 8.0))
    for a, (f, t) in zip(axes.ravel(), tiles):
        with PILImage.open(DATA_ROOT / f) as im:
            a.imshow(im.convert("RGB"))
        a.set_title(t, fontsize=8.5, color="#B22222" if t.startswith("WORST") else "#1B6E20")
        a.axis("off")
    for a in axes.ravel()[len(tiles):]:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_TH, dpi=110); plt.close()

# ---------------------------------------------------------------- styles (same as Steps 1-4)
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
    return [Paragraph(t, BL, bulletText="â€¢") for t in items]


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


def fig(path, w_cm, aspect, caption):
    return KeepTogether([Image(str(path), width=w_cm * cm, height=w_cm * cm * aspect),
                         Paragraph(caption, CAP)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "IDSE project - Solar panel soiling power-loss regression - Step 5 documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


gt, rt = comp[("grouped", "test")], comp[("random", "test")]
gv, rv = comp[("grouped", "val")], comp[("random", "val")]
pd21 = ev_day[("grouped", "test")].set_index("day").loc["2017-06-21"]
pd29 = ev_day[("grouped", "test")].set_index("day").loc["2017-06-29"]
pv = ev_day[("grouped", "val")].set_index("day")
rday = {sb: ev_day[("random", sb)] for sb in SETS}

# ---------------------------------------------------------------- content
s = []
s.append(Paragraph("Step 5: Evaluation and Error Analysis", ss["Title"]))
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
                   "Status: Step 5 <b>done</b>; Grad-CAM (Step 6) is in progress, robustness test next.",
                   CAP))

# summary
s.append(Paragraph("Summary of results", H1))
s.extend(bullets([
    f"<b>Headline (grouped test, 2 unseen days):</b> ResNet-18 MAE <b>{f4(gt['c']['MAE'])}</b> "
    f"({100 * gt['c']['MAE']:.2f} percentage points), RMSE {f4(gt['c']['RMSE'])}, {R2} "
    f"<b>{gt['c']['R2']:.3f}</b>, against the mean-predictor baseline MAE {f4(gt['b']['MAE'])} "
    f"({R2} {gt['b']['R2']:+.4f}): <b>{gt['red']:.1f}% lower MAE</b>.",
    f"<b>Random split (optimistic reference):</b> test MAE {f4(rt['c']['MAE'])}, {R2} "
    f"{rt['c']['R2']:.3f}. The grouped error is {gt['c']['MAE'] / rt['c']['MAE']:.2f}x larger, "
    "the cost of near-duplicate leakage measured in Step 2.",
    f"<b>Mean reversion:</b> the model over-predicts clean panels (grouped test low-L bias "
    f"{rng[('grouped', 'test', 'low')]['bias']:+.4f}) and under-predicts mid/high soiling "
    f"({rng[('grouped', 'test', 'mid')]['bias']:+.4f} / {rng[('grouped', 'test', 'high')]['bias']:+.4f}); "
    "the effect is about 2.6x stronger than on the random split for clean panels.",
    f"<b>Error is concentrated in days and sessions:</b> on grouped test, 21 Jun has MAE "
    f"{pd21.MAE:.4f} and 29 Jun {pd29.MAE:.4f}; 21 Jun carries "
    f"{share['test']['2017-06-21']:.0f}% of the total absolute error. On val, 30 Jun (heavy soiling) is "
    f"under-predicted by {abs(pv.loc['2017-06-30', 'bias']):.3f} on average.",
    f"<b>Worst cases are probably label/imaging anomalies:</b> all 20 worst grouped test predictions "
    f"are under-predictions (true L 0.83-0.99, predicted 0.04-0.16); they come from only {n_read} "
    f"distinct label readings, and the inspected images look visually clean.",
]))
s.append(Paragraph(f"All tables in this document were recomputed from the saved prediction files and "
                   f"agree with outputs/eval: {ok(all_ok)}.", CAP))

# 1 purpose
s.append(Paragraph("1. Purpose of Step 5 and where it fits", H1))
s.append(p(
    "Step 4 trained a pretrained ResNet-18 on both split strategies and saved, for the best checkpoint, "
    "one prediction per val/test image (" + C % "outputs/preds/*.csv" + "). Step 5 analyses these saved "
    "predictions. It does not retrain or re-run the network, so it needs no GPU and is fully "
    "reproducible in seconds. Its goals:"))
s.extend(bullets([
    "put the baseline and the CNN side by side for <b>every</b> split and set in one table;",
    "find <b>where</b> the model fails: by L range (the label is skewed), by day (the grouped unit), "
    "and in the individual worst predictions;",
    "measure systematic <b>bias</b> (mean of predicted minus true), not just the error size;",
    "produce the report figures (scatter plots, bar chart) and choose the failure cases for "
    "Grad-CAM (Step 6).",
]))
s.append(p("Inputs are read only; the raw images, split files and Step 4 outputs are not changed."))

# 2 what & why
s.append(Paragraph("2. What we did and why", H1))
s.append(table([
    ["#", "Action", "Why"],
    ["5.1", "Wrote " + C % "evaluate.py" + ": joins each prediction file with " +
     C % "splits/splits.csv" + " (day, irradiance I) and writes all tables and plots to " +
     C % "outputs/eval/" + ".",
     "One script, no torch, reproducible; predictions were saved in Step 4 so analysis never "
     "touches the model again (no risk of silently changing results)."],
    ["5.2", "Comparison table: baseline vs ResNet-18 (MAE, RMSE, " + R2 + ", MAE reduction, bias) for "
     "random/grouped x val/test.",
     "A single table that shows (a) the CNN learned something (vs baseline) and (b) how much the "
     "random split overstates performance (vs grouped)."],
    ["5.3", "Error and bias per L range (low &lt; 0.15, mid 0.15-0.5, high &gt;= 0.5) on test.",
     "L is right-skewed (median 0.12, mean 0.27); an overall MAE is dominated by clean panels and "
     "can hide failure on heavy soiling. Bias shows the direction of the error."],
    ["5.4", "Per-day error (n, MAE, bias, mean true L, mean predicted L) for all four sets.",
     "The day is the unit of the grouped split; the grouped test has only 2 days, so one bad day can "
     "dominate the headline number. The random per-day table is the contrast."],
    ["5.5", "Predicted-vs-true hexbin scatter plots (log count) for both test sets.",
     "Shows the full error structure (clusters, bands, outliers) that summary numbers hide; log "
     "colour because thousands of frames overlap."],
    ["5.6", "Bar chart of baseline vs ResNet-18 test MAE and RMSE.", "Report/slide figure."],
    ["5.7", "Worst-20 predictions per strategy (test) with day and I; visually inspected five "
     "grouped worst cases (one per session).",
     "Checks whether the biggest errors are model failures or label/imaging problems; picks the "
     "failure examples for Grad-CAM."],
], [0.9 * cm, 7.6 * cm, 8.5 * cm], small=True))

# 3 method
s.append(Paragraph("3. Method and definitions", H1))
s.append(Paragraph("3.1 Command", H2))
s.append(Preformatted(".\\.venv\\Scripts\\Activate.ps1\npython evaluate.py", CODE))
s.append(Paragraph("3.2 Inputs", H2))
s.extend(bullets([
    C % "outputs/preds/resnet18_{random,grouped}_{val,test}.csv" + " (filename, L_true, L_pred) "
    "from the best Step 4 checkpoints. L_pred is already clamped to [0, 1].",
    C % "splits/splits.csv" + ": adds day and I by filename, and gives the training-set mean of L per "
    f"strategy for the baseline (random {train_mean['random']:.4f}, grouped {train_mean['grouped']:.4f}). "
    f"Every prediction row found its split row: {ok(merge_ok)}.",
]))
s.append(Paragraph("3.3 Metrics (L as a fraction; x100 = percentage points)", H2))
s.append(table([
    ["Metric", "Definition", "Meaning here"],
    ["MAE", "mean |L_pred - L_true|", "Typical error size; easy to state (e.g. 8.69 points)."],
    ["RMSE", "sqrt(mean (L_pred - L_true)<super>2</super>)", "Punishes large errors; RMSE much larger "
     "than MAE means a few big misses."],
    [R2, "1 - SS<sub>res</sub> / SS<sub>tot</sub>, SS<sub>tot</sub> around the mean of the set "
     "itself", "0 = no better than the set's own mean; 1 = perfect. Can be negative."],
    ["bias", "mean (L_pred - L_true)", "Positive = over-predicts loss, negative = under-predicts."],
    ["MAE reduction", "100 (1 - MAE<sub>CNN</sub> / MAE<sub>baseline</sub>)", "How much of the "
     "baseline error the image removes."],
    ["Baseline", "predict the train-mean L of the same strategy for every image", "Same as Step 3."],
], [2.4 * cm, 6.3 * cm, 8.3 * cm], small=True))

# 4 overall comparison
s.append(Paragraph("4. Baseline vs ResNet-18 on every set", H1))
rows = [["Split", "Set", "n", "Base MAE", "Base RMSE", "Base " + R2, "CNN MAE", "CNN RMSE",
         "CNN " + R2, "MAE red.", "Bias"]]
shade = {}
for st in STRATS:
    for sb in SETS:
        v = comp[(st, sb)]
        rows.append([st, sb, f"{v['n']:,}", f4(v["b"]["MAE"]), f4(v["b"]["RMSE"]),
                     f"{v['b']['R2']:+.4f}", f4(v["c"]["MAE"]), f4(v["c"]["RMSE"]),
                     f"{v['c']['R2']:.4f}", f"{v['red']:.1f}%", f"{v['bias']:+.4f}"])
        if st == "grouped" and sb == "test":
            shade[len(rows) - 1] = "#E8F4EA"
s.append(table(rows, [1.5 * cm, 1 * cm, 1.3 * cm, 1.5 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm, 1.6 * cm,
                      1.5 * cm, 1.5 * cm, 1.5 * cm], small=True, shade_rows=shade))
s.append(Paragraph("Green: headline result (grouped test, the only fully held-out estimate). "
                   "Source: outputs/eval/comparison_all.csv.", CAP))
s.append(Spacer(1, 4))
s.append(fig(EV / "comparison_bar.png", 12, 480 / 840,
             "Figure 1. Test MAE and RMSE, mean baseline vs ResNet-18 (outputs/eval/comparison_bar.png)."))
s.append(Paragraph("4.1 Reading the table", H2))
s.extend(bullets([
    f"The CNN beats the baseline on all four sets, by {gt['red']:.1f}% (grouped test) to "
    f"{rv['red']:.1f}% (random val) in MAE. The image carries real information about L.",
    f"<b>Random val and test are almost identical</b> (MAE {f4(rv['c']['MAE'])} vs "
    f"{f4(rt['c']['MAE'])}, {R2} {rv['c']['R2']:.4f} vs {rt['c']['R2']:.4f}) because both are drawn "
    "from the same days as train. <b>Grouped val and test differ</b> (MAE "
    f"{f4(gv['c']['MAE'])} vs {f4(gt['c']['MAE'])}, {R2} {gv['c']['R2']:.3f} vs {gt['c']['R2']:.3f}) "
    "because they are different days with different conditions.",
    f"Grouped test {R2} ({gt['c']['R2']:.3f}) is lower than grouped val ({gv['c']['R2']:.3f}) even "
    "though the test MAE is only slightly higher: the test days have a narrower L distribution "
    "(std 0.245 vs 0.273, Step 2), so SS<sub>tot</sub> is smaller and the same errors explain less "
    f"of the variance. RMSE/MAE is {gt['c']['RMSE'] / gt['c']['MAE']:.2f} on grouped test vs "
    f"{gv['c']['RMSE'] / gv['c']['MAE']:.2f} on grouped val, i.e. the test error contains more "
    "large misses (Section 7).",
    f"Grouped baseline {R2} on val is {gv['b']['R2']:+.4f}: the grouped val days have a lower mean L "
    "(0.216) than the train mean (0.272), so a constant train mean is worse than the val mean.",
    "Overall bias is small on every set (|bias| &lt; 0.017), but this hides large opposite biases "
    "per L range and per day (Sections 5 and 6).",
]))

# 5 by L range
s.append(Paragraph("5. Error and bias by L range", H1))
rows = [["Split", "Set", "Range", "n", "Baseline MAE", "CNN MAE", "CNN bias"]]
shade = {}
names = {"low": "low (L &lt; 0.15)", "mid": "mid (0.15-0.5)", "high": "high (L &gt;= 0.5)"}
for st in STRATS:
    for sb in SETS:
        for _, _, k in BINS:
            v = rng[(st, sb, k)]
            rows.append([st, sb, names[k], f"{v['n']:,}", f4(v["base"]), f4(v["MAE"]),
                         f"{v['bias']:+.4f}"])
            if v["MAE"] > v["base"]:
                shade[len(rows) - 1] = "#FCE8E6"
s.append(table(rows, [1.8 * cm, 1.2 * cm, 3.4 * cm, 1.6 * cm, 2.8 * cm, 2.6 * cm, 2.6 * cm],
               small=True, shade_rows=shade))
s.append(Paragraph("Test rows are the content of outputs/eval/error_by_range.csv; val rows are added "
                   "here from the same prediction files. Red: CNN MAE above baseline MAE in that range.",
                   CAP))
s.append(Paragraph("5.1 Findings", H2))
s.extend(bullets([
    "<b>Regression toward the mean.</b> On every set the bias is positive for clean panels and "
    "negative for mid and high soiling: the model pulls predictions towards the middle of the "
    "distribution. This is the typical behaviour of an MSE-trained regressor on noisy labels: when "
    "the image is ambiguous, predicting near the mean minimises the expected squared error.",
    f"<b>Much stronger on the grouped split.</b> Clean-panel bias on test is "
    f"{rng[('grouped', 'test', 'low')]['bias']:+.4f} (grouped) vs "
    f"{rng[('random', 'test', 'low')]['bias']:+.4f} (random), and the clean-panel MAE doubles "
    f"({f4(rng[('random', 'test', 'low')]['MAE'])} to {f4(rng[('grouped', 'test', 'low')]['MAE'])}). "
    "With only 10 training days, the model is less sure on a new day and hedges more.",
    f"<b>Grouped test low-L error is almost all bias:</b> MAE {f4(rng[('grouped', 'test', 'low')]['MAE'])} "
    f"with bias {rng[('grouped', 'test', 'low')]['bias']:+.4f}, so nearly every clean test frame is "
    "predicted too high, by a similar amount (a systematic offset, not random scatter). "
    f"{r21['low'][0]:,} of the {rng[('grouped', 'test', 'low')]['n']:,} clean test frames are from "
    "21 Jun (Section 6).",
    f"<b>Mid range on the grouped split is the hardest</b>: CNN MAE "
    f"{f4(rng[('grouped', 'test', 'mid')]['MAE'])} vs baseline "
    f"{f4(rng[('grouped', 'test', 'mid')]['base'])} (test) - the constant ~0.27 sits inside this range, "
    "so the baseline is naturally strong there (already noted in Step 4).",
    f"<b>Heavy soiling on grouped val</b> is the weakest range (MAE "
    f"{f4(rng[('grouped', 'val', 'high')]['MAE'])}, bias {rng[('grouped', 'val', 'high')]['bias']:+.4f}): "
    "practically every heavy val image is under-predicted. The grouped test set has almost no L "
    "above 0.8, so very heavy soiling is judged mainly on val.",
]))

# 6 per day
s.append(Paragraph("6. Per-day error", H1))
s.append(Paragraph("6.1 Grouped split (val and test days)", H2))
rows = [["Set", "Day", "n", "MAE", "Bias", "Mean true L", "Mean pred. L", "Share of set's abs. error"]]
for sb in SETS:
    e = ev_day[("grouped", sb)]
    for _, r in e.iterrows():
        rows.append([sb, r.day, f"{int(r.n):,}", f4(r.MAE), f"{r.bias:+.4f}", f4(r.L_true_mean),
                     f4(r.L_pred_mean), f"{share[sb][r.day]:.1f}%"])
s.append(table(rows, [1.2 * cm, 2.3 * cm, 1.4 * cm, 1.7 * cm, 1.8 * cm, 2.2 * cm, 2.3 * cm, 4.1 * cm],
               small=True, shade_rows={4: "#FCE8E6", 3: "#FCE8E6"}))
s.append(Paragraph("Source: outputs/eval/per_day_error_grouped_{val,test}.csv; the last column is "
                   "computed here from the prediction files.", CAP))
s.append(fig(FIG_DAY, 17, 4.2 / 13, "Figure 2. Grouped split per-day MAE (left) and mean true vs "
             "predicted L (right). Light bars: val days; dark bars: test days."))
s.append(Paragraph("6.2 Findings", H2))
s.extend(bullets([
    f"<b>The grouped test error is dominated by 21 Jun</b>: MAE {pd21.MAE:.4f} on 4,777 frames vs "
    f"{pd29.MAE:.4f} on 1,956 frames for 29 Jun. 21 Jun contributes "
    f"{share['test']['2017-06-21']:.0f}% of all absolute error on the test set. Without 21 Jun the "
    f"test MAE would be {f4(excl['2017-06-21']['MAE'])}; without 29 Jun it would be "
    f"{f4(excl['2017-06-29']['MAE'])} (computed here, for illustration only - the test set is not "
    "changed).",
    f"21 Jun has a near-zero mean bias ({pd21.bias:+.4f}) but a high MAE: large errors in both "
    f"directions cancel. Per range on 21 Jun: low n={r21['low'][0]:,} MAE {r21['low'][1]:.4f} "
    f"(bias {r21['low'][2]:+.4f}); mid n={r21['mid'][0]:,} MAE {r21['mid'][1]:.4f} (bias "
    f"{r21['mid'][2]:+.4f}); high n={r21['high'][0]:,} MAE {r21['high'][1]:.4f} (bias "
    f"{r21['high'][2]:+.4f}). The many clean frames are over-predicted by ~0.08, and the few "
    "frames labelled heavily soiled are badly under-predicted (they look clean to the model, see "
    "Figure 3 and Section 8).",
    f"29 Jun (true mean {pd29.L_true_mean:.3f}) is predicted well on average ({pd29.L_pred_mean:.3f}); "
    "its error comes from a few short sessions with near-total loss labels (Section 8).",
    f"<b>Val:</b> 13 Jun is nearly unbiased (MAE {pv.loc['2017-06-13', 'MAE']:.4f}). 25 Jun, the "
    f"cleanest day (true mean {pv.loc['2017-06-25', 'L_true_mean']:.3f}), is over-predicted "
    f"({pv.loc['2017-06-25', 'L_pred_mean']:.3f}). 30 Jun, the most soiled day (true mean "
    f"{pv.loc['2017-06-30', 'L_true_mean']:.3f}), is under-predicted "
    f"({pv.loc['2017-06-30', 'L_pred_mean']:.3f}) and has the largest MAE of any day "
    f"({pv.loc['2017-06-30', 'MAE']:.4f}). The day-level picture is the same mean reversion as in "
    "Section 5: extreme days are pulled towards the middle.",
]))
s.append(fig(FIG_TL, 17, 6.6 / 13, "Figure 3. Grouped test days over time: true L (black), predicted L "
             "(red) and irradiance I (green, right axis). Made for this document from the saved "
             "predictions."))
s.append(p("Figure 3 (made for this document) shows the time structure behind the per-day numbers:"))
s.extend(bullets([
    "<b>29 Jun:</b> the label sits at ~0.65 all morning and the prediction follows it closely. "
    "Around 12:25 the true L drops abruptly to ~0.02 and stays near 0 for the rest of the day, "
    "consistent with the panel being cleaned at that time (an inference from the labels; no cleaning "
    "log is available). The prediction drops with it to ~0.1. The 12:27-12:28 worst cases (true "
    "0.83-0.88) sit exactly at this transition, and the 13:31 / 13:49 cases (true ~0.99) are "
    "isolated spikes inside the clean period: the image already shows a clean panel while the label "
    "is high, which points to a label that lags or glitches rather than to the model.",
    "<b>21 Jun:</b> from about 10:15 to 17:00 the panel is clean (true L mostly 0-0.1) and the "
    "prediction stays slightly high (~0.1), the over-prediction seen in Section 5. In the same hours "
    "the irradiance (green) swings rapidly, as with passing clouds, and the true L shows many "
    "scattered isolated spikes up to 0.6-1.0 that the prediction does not follow. These spikes are "
    "much more frequent while the irradiance is fluctuating, which suggests (not proven) that the "
    "loss label is sensitive to fast light changes, e.g. the panel and the reference not being "
    "measured at exactly the same moment.",
    "<b>21 Jun morning (6-10 h):</b> the label is at 0.1-0.4 and the prediction tracks the level but "
    "with offsets (e.g. predicted ~0.35 vs true ~0.15-0.25 just after 6 h, and a jump to ~0.65 near "
    "9:45 while the label is ~0.45). <b>21 Jun evening (after 17:45):</b> a few frames are labelled "
    "0.6-0.93 at irradiance below ~0.05, where a loss ratio is numerically unreliable.",
]))
s.append(Paragraph("6.3 Random split per day (contrast)", H2))
rt_d, rv_d = rday["test"], rday["val"]
s.append(p(
    f"On the random split every one of the 15 days appears in val and test. Per-day test MAE ranges "
    f"only from {rt_d.MAE.min():.4f} ({rt_d.loc[rt_d.MAE.idxmin(), 'day']}) to {rt_d.MAE.max():.4f} "
    f"({rt_d.loc[rt_d.MAE.idxmax(), 'day']}), with |bias| at most {rt_d.bias.abs().max():.4f} per day "
    f"(val: {rv_d.MAE.min():.4f} to {rv_d.MAE.max():.4f}). The same day 21 Jun that gives MAE "
    f"{pd21.MAE:.4f} as an unseen grouped test day gives only "
    f"{rt_d.set_index('day').loc['2017-06-21', 'MAE']:.4f} on the random split, where its neighbouring "
    "frames were in training. 30 Jun: "
    f"{pv.loc['2017-06-30', 'MAE']:.4f} unseen (grouped val) vs "
    f"{rt_d.set_index('day').loc['2017-06-30', 'MAE']:.4f} (random test). This is the leakage effect, "
    "seen day by day."))
rows = [["Day", "n (test)", "MAE", "Bias", "Mean true L", "Mean pred. L"]]
for _, r in rt_d.iterrows():
    rows.append([r.day, int(r.n), f4(r.MAE), f"{r.bias:+.4f}", f4(r.L_true_mean), f4(r.L_pred_mean)])
s.append(table(rows, [3 * cm, 2 * cm, 2.4 * cm, 2.4 * cm, 2.8 * cm, 2.8 * cm], small=True))
s.append(Paragraph("Random split, test set, per day (outputs/eval/per_day_error_random_test.csv; the "
                   "val file has the same structure).", CAP))

# 7 scatter
s.append(Paragraph("7. Predicted vs true L (scatter plots)", H1))
sc = Table([[Image(str(EV / "scatter_random_test.png"), width=8.2 * cm, height=8.2 * cm),
             Image(str(EV / "scatter_grouped_test.png"), width=8.2 * cm, height=8.2 * cm)]],
           colWidths=[8.5 * cm, 8.5 * cm])
s.append(KeepTogether([sc, Paragraph(
    "Figure 4. Hexbin of predicted vs true L on the test sets (colour = log10 count; red dashed line = "
    "perfect prediction). Left: random split; right: grouped split (outputs/eval/scatter_*_test.png).",
    CAP)]))
s.extend(bullets([
    "<b>Random split:</b> points hug the diagonal tightly over the whole range 0-1.",
    "<b>Grouped split</b> shows distinct structures:",
    "&nbsp;&nbsp;- a correct, tight cluster at true L ~0.65 predicted ~0.65-0.7 (mostly 29 Jun);",
    "&nbsp;&nbsp;- a clean-panel cluster (true ~0-0.05) predicted slightly high (~0.1): the positive "
    "low-L bias;",
    "&nbsp;&nbsp;- a band of true L 0.15-0.45 predicted too low (~0-0.15): moderately soiled frames "
    "that look clean to the model;",
    f"&nbsp;&nbsp;- a few points with true L 0.8-0.9+ predicted ~0.1: the worst failures. Of the "
    f"{n_true_hi:,} grouped test frames with true L &gt;= 0.8, {n_true_hi_pred_lo:,} are predicted "
    f"below 0.2. Overall {n_big:,} frames ({100 * n_big / gt['n']:.1f}%) have |error| &gt; 0.5, "
    f"{n_big_under:,} of them under-predictions.",
    f"Predictions come out in discrete steps ({n_unique_pred:,} distinct values on grouped test) "
    "because the network ran in bf16; the step is about 0.002 near L = 0.3, far below the errors "
    f"discussed here. {100 * frac0:.1f}% of grouped test predictions were clamped to exactly 0.",
]))

# 8 worst cases
s.append(Paragraph("8. Worst-20 predictions and visual inspection", H1))
s.append(Paragraph("8.1 Grouped test", H2))
rows = [["#", "Day / time", "True L", "Pred. L", "Error", "I"]]
for i, r in enumerate(wg.sort_values("err").itertuples(), 1):
    rows.append([i, f"{r.day[-2:]} Jun {r.timestamp:%H:%M:%S}", f4(r.L_true), f4(r.L_pred),
                 f"{r.err:+.4f}", f4(r.I)])
s.append(table(rows, [0.8 * cm, 3.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm], small=True))
s.append(Paragraph("Source: outputs/eval/worst20_grouped_test.csv (sorted by |error|).", CAP))
s.extend(bullets([
    "<b>All 20 are under-predictions</b> (error -0.93 to -0.70): true L 0.83-0.99 (almost total "
    "loss), predicted 0.04-0.16 (almost clean).",
    f"They come from only <b>{n_read} distinct (L, I) label readings</b> in {n_win} short time "
    f"windows (frames less than 5 min apart grouped together; these windows lie in {n_sess} of the "
    "57 recording sessions of Step 2, one per test day). Consecutive frames share identical L and I, "
    "i.e. the label is updated more slowly than the ~5 s image cadence, so a handful of readings "
    "account for many of the worst cases.",
    "Time windows: 29 Jun 13:31 and 13:49 (true L 0.870-0.991, I ~0.34-0.38, predicted 0.068-0.114); "
    "29 Jun 12:27:54-12:28:09 (true L 0.827-0.881, I 0.59-0.66, predicted 0.13-0.15); 21 Jun 13:15:13 "
    "(true L 0.9895, I 0.386, predicted 0.161); 21 Jun 17:49-17:58 evening frames (true L "
    "0.860-0.928, I only 0.027-0.052, predicted 0.044-0.096).",
]))
s.append(Paragraph("8.2 Visual inspection", H2))
s.append(p(
    "Five of these images (one per session listed above) were opened and looked at during Step 5; "
    f"Figure 5 shows one frame from each of the {n_win} time windows. In all of them "
    "the panel looks <b>visually clean</b>: no white deposits or streaks like those on correctly "
    "predicted soiled frames. The label says almost total loss while nothing visible in the image "
    "explains it. The first inspection also noted that the framing looked tighter than usual (panel "
    "running off the right edge); next to the reference frames in Figure 5 the difference is small, "
    "so framing is <b>not</b> treated as a confirmed factor."))
if img_ok:
    s.append(fig(FIG_TH, 17, 8.0 / 13,
                 f"Figure 5. First {len(picks)} images: the first frame of each of the {n_win} "
                 "worst-case time windows (red titles). Last two: correctly predicted grouped test frames for "
                 "reference (a ~0.65 soiled frame and a clean frame, green titles), chosen at random "
                 "(seed 0) among frames with small error."))
else:
    s.append(Paragraph("(Thumbnail figure skipped: raw images not found at DATA_ROOT.)", CAP))
s.append(p(
    "<b>Interpretation (hypothesis, not verified):</b> these are label or imaging anomalies (e.g. an "
    "electrical/measurement event, partial shading or a sensor glitch that the image cannot show, or a "
    "camera/framing change), not a pure model failure. A model that judges the visible panel "
    "<i>should</i> predict low L for these frames. The 21 Jun evening frames also have very low "
    "irradiance (I 0.027-0.052), where a loss ratio is numerically unreliable - consistent with the "
    "concern raised in Step 1 about I &lt; 0.03. Because MSE squares these ~0.9 errors, these few "
    "readings inflate the grouped test RMSE far more than the MAE."))
s.append(Paragraph("8.3 Random test (contrast)", H2))
wr = w20["random"]
s.append(p(
    f"The random worst-20 is more mixed: {int((wr.err < 0).sum())} under-predictions and "
    f"{int((wr.err > 0).sum())} over-predictions, max |error| {wr.err.abs().max():.4f} (grouped: "
    f"{w20['grouped'].err.abs().max():.4f}), spread over {wr.day.nunique()} days. Examples: "
    "24 Jun 13:54 (true 0.995, predicted 0.178); a 22 Jun 15:31 session (true 0.81-0.82, predicted "
    "0.04-0.08); and over-predictions of clean frames in poor light, e.g. 27 Jun 07:05 (true 0.007, "
    "predicted 0.727, I 0.082) and 29 Jun 06:02 (true 0.040, predicted 0.566, I 0.003). So the same "
    "kind of 'label does not match the picture' frames exist across the whole dataset, not only on "
    "the grouped test days."))

# 9 low irradiance
s.append(Paragraph("9. Sensitivity check: low-irradiance frames (I &lt; 0.1)", H1))
s.append(p(
    "Step 1 and the worst-case list both point to unreliable labels when irradiance is very low. As a "
    "quick check, the metrics were recomputed here <b>from the saved predictions</b> after excluding "
    "frames with I &lt; 0.1. This is a sensitivity analysis only: the model, the checkpoint and the "
    "official test set are unchanged, and the headline numbers stay the full-set numbers."))
rows = [["Split", "Set", "Frames with I &lt; 0.1", "MAE on those frames", "MAE (I &gt;= 0.1)",
         "RMSE (I &gt;= 0.1)", R2 + " (I &gt;= 0.1)", "Full-set MAE"]]
for st in STRATS:
    for sb in SETS:
        v = lowI[(st, sb)]
        rows.append([st, sb, f"{v['n_low']:,} ({100 * v['frac']:.1f}%)", f4(v["MAE_lowI"]),
                     f4(v["MAE"]), f4(v["RMSE"]), f"{v['R2']:.4f}", f4(comp[(st, sb)]["c"]["MAE"])])
s.append(table(rows, [1.5 * cm, 1 * cm, 2.6 * cm, 2.3 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.2 * cm],
               small=True))
lg = lowI[("grouped", "test")]
s.append(Spacer(1, 3))
s.append(p(
    f"On grouped test, frames with I &lt; 0.1 have MAE {f4(lg['MAE_lowI'])} vs {f4(lg['MAE'])} on the "
    f"rest; excluding them changes the grouped test MAE from {f4(gt['c']['MAE'])} to {f4(lg['MAE'])} "
    f"and {R2} from {gt['c']['R2']:.3f} to {lg['R2']:.3f}. On the other three sets the low-light frames "
    "are <b>not</b> worse than the rest (e.g. grouped val "
    f"{f4(lowI[('grouped', 'val')]['MAE_lowI'])} vs {f4(lowI[('grouped', 'val')]['MAE'])}), so the "
    "effect is specific to the grouped test days and small overall. This supports (but does not prove) "
    "that part of the grouped test error comes from unreliable low-light labels. It can be mentioned as a limitation and a "
    "possible data-cleaning rule for future work; it is not used to replace the headline result."))

# 10 conclusions
s.append(Paragraph("10. Conclusions for the report", H1))
s.extend(bullets([
    f"<b>Headline generalisation result = grouped test:</b> MAE {f4(gt['c']['MAE'])} "
    f"({100 * gt['c']['MAE']:.2f} pts), RMSE {f4(gt['c']['RMSE'])}, {R2} {gt['c']['R2']:.3f}, vs "
    f"baseline MAE {f4(gt['b']['MAE'])} and {R2} about 0. The model explains about "
    f"{100 * gt['c']['R2']:.0f}% of the variance of L on two days it never saw.",
    f"<b>The random split is optimistic</b> (MAE {f4(rt['c']['MAE'])}, {R2} {rt['c']['R2']:.3f}) because "
    "of near-duplicate frames 5 s apart; it is shown only as a reference for the leakage effect.",
    "<b>Errors are not uniform:</b> they concentrate on a few days (21 Jun on test, 30 Jun on val) and "
    "a few sessions whose labels do not match the visible panel state.",
    "<b>Systematic mean-reversion bias:</b> clean panels over-predicted, heavy soiling under-predicted, "
    "much stronger for unseen days.",
    "<b>Limits of the evidence:</b> the test set is 2 days, one seed, one site and one camera; the "
    "anomaly explanation for the worst cases is a hypothesis.",
]))

# 11 outputs
s.append(Paragraph("11. Outputs produced", H1))
s.append(table([
    ["File", "Content"],
    ["evaluate.py", "Step 5 script (no torch needed)."],
    ["outputs/eval/comparison_all.csv", "Baseline vs CNN, all 4 split/set combinations (Section 4)."],
    ["outputs/eval/comparison_test.csv", "Same, test rows only (report table)."],
    ["outputs/eval/error_by_range.csv", "n, MAE, bias per L range, test sets (Section 5)."],
    ["outputs/eval/per_day_error_{random,grouped}_{val,test}.csv",
     "n, MAE, bias, mean true/predicted L per day (Section 6)."],
    ["outputs/eval/worst20_{random,grouped}_test.csv",
     "20 largest |errors|: filename, day, L_true, L_pred, err, I (Section 8)."],
    ["outputs/eval/scatter_{random,grouped}_test.png", "Hexbin scatter plots (Figure 4)."],
    ["outputs/eval/comparison_bar.png", "Baseline vs CNN bar chart (Figure 1)."],
    ["docs/step5_figs/", "Figures 2, 3, 5 of this document (made by docs/build_step5_pdf.py from the "
                         "saved predictions)."],
], [7.4 * cm, 9.6 * cm], small=True))

# 12 decisions and caveats
s.append(Paragraph("12. Decisions and caveats", H1))
s.extend(bullets([
    "<b>No re-evaluation of the network:</b> all analysis uses the predictions saved in Step 4, so the "
    "numbers are exactly those of the selected checkpoints and match Step 4.",
    "<b>Test was not used to change anything.</b> The checkpoint was selected on val in Step 4; "
    "grouped val numbers are therefore slightly optimistic and only the grouped test is truly "
    "held out. The per-day exclusions (Section 6) and the I &lt; 0.1 check (Section 9) are "
    "diagnostics, not a new headline.",
    "<b>Only 2 grouped test days</b> (21 and 29 Jun): one bad day moves the headline number a lot "
    "(MAE without 21 Jun would be " + f4(excl["2017-06-21"]["MAE"]) + "). Single seed, one site, "
    "one camera.",
    "<b>Worst-case explanation is a hypothesis.</b> Five images were inspected by eye; no sensor logs "
    "or extra metadata are available to confirm a label or imaging anomaly.",
    "<b>Consecutive frames are not independent:</b> many frames share the same label reading, so "
    "the effective number of independent test samples is much smaller than 6,733; error counts "
    "should be read with that in mind.",
    "<b>Per-range table in evaluate.py covers test only</b>; the val rows in Section 5 were added "
    "here from the same files. The docstring of evaluate.py also mentions an " +
    C % "error_by_range.png" + " that the script does not write (only the CSV exists).",
    "<b>Bias sign convention</b> is predicted minus true throughout (negative = under-prediction).",
]))

# 13 next
s.append(Paragraph("13. Next steps", H1))
s.extend(bullets([
    "<b>Step 6 - Grad-CAM</b> (" + C % "gradcam.py" + ", in progress): clean / moderate / heavy "
    "correct predictions plus the worst failures from Section 8 (one per distinct label value). "
    "Check that the model attends to soiled regions of the panel and not to the background.",
    "<b>Robustness test</b> on the grouped test set: brightness, Gaussian noise, JPEG compression; "
    "MAE/RMSE/" + R2 + " before vs after.",
    "Optional: EfficientNet-B0 comparison if time allows. Then report, slides and viva preparation.",
]))
s.append(KeepTogether([Paragraph("13.1 Viva points from this step", H2), table([
    ["Question", "Answer grounded in Step 5"],
    ["Why is grouped test worse than random test?",
     f"Leakage: random test frames have a training frame ~5 s away. Day by day, 21 Jun has MAE "
     f"{pd21.MAE:.3f} unseen vs {rt_d.set_index('day').loc['2017-06-21', 'MAE']:.3f} on the random split."],
    ["Why over-predict clean and under-predict heavy?",
     "Regression toward the mean under MSE with noisy labels and only 10 training days."],
    ["Are the worst errors model failures?",
     f"Probably mostly not: {n_read} label readings, visually clean panels labelled ~0.9-0.99 loss, "
     "some at very low irradiance. Stated as a hypothesis."],
    ["Why is R<super>2</super> lower on grouped test than val with similar MAE?",
     "Narrower L spread on the test days (smaller total variance) and a few very large misses."],
], [6 * cm, 11 * cm], small=True)]))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Step 5: Evaluation and Error Analysis",
                        author="IDSE Solar Soiling Team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("eval CSVs match recomputation:", all_ok, "| merge ok:", merge_ok, "| images found:", img_ok)
print("share of abs error (test):", share["test"].round(1).to_dict())
print("excl 21 Jun:", {k: round(v, 4) for k, v in excl["2017-06-21"].items()})
print("21 Jun by range:", r21)
print("worst20: readings", n_read, "sessions", n_sess)
print("big errors:", n_big, n_big_under, "| true>=0.8:", n_true_hi, "pred<0.2:", n_true_hi_pred_lo)
print("lowI:", {k: {m: round(float(x), 4) for m, x in v.items()} for k, v in lowI.items()})
print("unique preds:", n_unique_pred, "clamped0:", round(frac0, 4))
print("Wrote", PDF)
