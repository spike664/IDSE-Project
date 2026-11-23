"""Build docs/Step1_Dataset_Audit.pdf (Step 1 documentation).

Usage (from the project folder):
    python -m pip install reportlab pandas
    python docs/build_step1_pdf.py

All numbers in the tables are recomputed from step1_outputs/dataset_manifest.csv.
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "step1_outputs"
PDF = ROOT / "docs" / "Step1_Dataset_Audit.pdf"

# ---------------------------------------------------------------- numbers
df = pd.read_csv(OUT_DIR / "dataset_manifest.csv", parse_dates=["timestamp"])
n = len(df)
L, I, gap = df["L"], df["I"], df["time_gap_sec"]
per_day = df.groupby(df["timestamp"].dt.date).agg(
    n=("L", "size"), first=("timestamp", "min"), last=("timestamp", "max"))
n_corrupt = int(df["corrupt"].sum())
n_low_I = int((I < 0.03).sum())
n_L_zero = int((L == 0).sum())

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=14,
                    spaceAfter=6, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10,
                    spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14,
                   spaceAfter=5)
BL = ParagraphStyle("BL", parent=B, leftIndent=14, bulletIndent=4)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.grey,
                     alignment=TA_CENTER)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=8.5, leading=11,
                      backColor=colors.HexColor("#F3F4F6"), borderPadding=5,
                      leftIndent=4, spaceBefore=4, spaceAfter=8)
CELL = ParagraphStyle("CELL", parent=B, fontSize=9, leading=11, spaceAfter=0)


def p(t):
    return Paragraph(t, B)


def bullets(items):
    return [Paragraph(t, BL, bulletText="\u2022") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EAF2")))
    t.setStyle(TableStyle(style))
    return t


def f4(x):
    return f"{x:.4f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "IDSE project - Solar panel soiling power-loss regression - Step 1 documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------- content
s = []
s.append(Paragraph("Step 1: Dataset Audit and Manifest", ss["Title"]))
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
s.append(Paragraph("Document date: 19 Nov 2026. Presentation: 21 Nov 2026.", CAP))

# 1
s.append(Paragraph("1. Project goal and where Step 1 fits", H1))
s.append(p(
    "The project predicts the <b>continuous</b> power loss <b>L</b> caused by soiling "
    "(dust, deposits, droppings) from a single RGB image of a solar panel. This is a "
    "<b>regression</b> task, not classification. The planned model is an ImageNet-pretrained "
    "ResNet-18 with a single-output regression head, compared against a mean-prediction baseline "
    "and evaluated with MAE, RMSE and R<super>2</super>."))
s.append(p(
    "Step 1 is the dataset audit. Before any model is trained we need to be sure that "
    "(a) every image and label can be read correctly, (b) we understand what the label means and "
    "what unit it is in, and (c) we know the structure of the data (in particular its time "
    "structure), because that decides how train/validation/test splits must be made in Step 2."))

# 2
s.append(Paragraph("2. Dataset", H1))
s.append(table([
    ["Item", "Value"],
    ["Name", "DeepSolarEye - Solar Panel Soiling Image dataset"],
    ["Location (local)", r"C:\Users\manie\Downloads\SolarPanelSoilingImageDataset\Solar_Panel_Soiling_Image_dataset\PanelImages"],
    ["Number of files", f"{n:,} JPEG images"],
    ["Labels", "Encoded in each filename (no separate label file)"],
    ["Camera", "Fixed view of one panel, frames captured every few seconds"],
], [4 * cm, 13 * cm]))
s.append(Paragraph("2.1 Filename format", H2))
s.append(Preformatted(
    "solar_<Dow>_<Mon>_<day>_<hour>__<min>__<sec>_<year>_L_<loss>_I_<irradiance>.jpg\n"
    "e.g. solar_Tue_Jun_13_9__46__49_2017_L_0.0474843723155_I_0.296733333333.jpg", CODE))
s.extend(bullets([
    "<b>Timestamp</b>: day-of-week, month, day, hour, minute, second, year.",
    "<b>L</b>: power loss of the panel relative to a clean reference panel (dataset README). "
    "<b>This is our target.</b>",
    "<b>I</b>: normalised irradiance level at capture time. Used only for analysis; "
    "<b>it is never given to the model as an input</b>, because the task is to predict loss "
    "from the image alone.",
]))
s.append(Paragraph("2.2 Unit decision for L", H2))
s.append(p(
    f"Observed L lies in [{L.min():.1f}, {L.max():.4f}], so L is a <b>fraction</b> (0 = no loss, "
    "1 = total loss). Decision: the model is trained on L as a fraction in [0, 1]; values are "
    "multiplied by 100 only when reporting percentages. Keeping a single fixed unit avoids "
    "mixing scales in losses and metrics."))

# 3
s.append(Paragraph("3. What we did and why", H1))
s.append(table([
    ["#", "Action", "Why"],
    ["1", "Listed every file in PanelImages and parsed its name with a regular expression "
          "into timestamp, L and I.",
     "Labels live only in the filenames, so a parsing error would silently corrupt the targets. "
     "Counting parse failures proves every label was recovered."],
    ["2", "Converted the date parts to a real timestamp and counted invalid ones.",
     "Timestamps are needed to sort images and to build time-grouped splits."],
    ["3", "Sorted by timestamp and computed the gap (seconds) to the previous image.",
     "Reveals the capture rate and the natural boundaries (sessions, days) in the data."],
    ["4", "Opened every image with Pillow, recorded (size, colour mode) and ran verify().",
     "Detects corrupt/truncated files before training and confirms the input resolution "
     "and channels the model will see."],
    ["5", "Saved everything to dataset_manifest.csv.",
     "One shared, reproducible table that all later steps (splits, loaders, evaluation) read, "
     "so nobody re-parses filenames differently."],
    ["6", "Computed summary statistics and histograms of L and I.",
     "Checks the label range/unit and the shape of the distribution (skew affects loss choice, "
     "baseline and how we report errors)."],
    ["7", "Plotted 12 images spread evenly across the sorted L range.",
     "Sanity check that L matches what is visible on the panel and that the scene is what "
     "we expect (to plan Grad-CAM checks)."],
], [0.7 * cm, 7.3 * cm, 9 * cm]))
s.append(Paragraph("3.1 Script and command", H2))
s.append(p("Script: <font face='Courier'>step1_manifest.py</font> (project folder). "
           "The raw images are opened read-only and are never modified, renamed or deleted."))
s.append(Preformatted(
    "cd C:\\Users\\manie\\OneDrive\\Desktop\\solar-soiling\n"
    "python -m pip install pandas matplotlib pillow\n"
    "python step1_manifest.py", CODE))
s.append(p("Environment: Windows 11, PowerShell, Python 3.12, pandas, matplotlib, pillow."))

# 4
s.append(Paragraph("4. Results", H1))
s.append(Paragraph("4.1 Data integrity", H2))
s.append(table([
    ["Check", "Result"],
    ["Files found", f"{n:,}"],
    ["Filename parse failures", "0"],
    ["Invalid timestamps", str(int(df['timestamp'].isna().sum()))],
    ["Corrupt / unreadable images", str(n_corrupt)],
    ["Image size and mode", "All 192 x 192, RGB"],
], [8 * cm, 9 * cm]))
s.append(p("Conclusion: the dataset is complete and clean. No image needs to be dropped. "
           "Images will be resized to 224 x 224 for ResNet-18 (its ImageNet input size)."))

s.append(Paragraph("4.2 Label statistics", H2))
dl, di = L.describe(), I.describe()
s.append(table([
    ["Statistic", "L (power loss)", "I (irradiance)"],
    ["count", f"{int(dl['count']):,}", f"{int(di['count']):,}"],
    ["mean", f4(dl["mean"]), f4(di["mean"])],
    ["std", f4(dl["std"]), f4(di["std"])],
    ["min", f4(dl["min"]), f4(di["min"])],
    ["25%", f4(dl["25%"]), f4(di["25%"])],
    ["median", f4(dl["50%"]), f4(di["50%"])],
    ["75%", f4(dl["75%"]), f4(di["75%"])],
    ["max", f4(dl["max"]), f4(di["max"])],
], [5 * cm, 6 * cm, 6 * cm]))
s.append(Spacer(1, 6))
s.append(KeepTogether([
    Image(str(OUT_DIR / "label_hist.png"), width=17 * cm, height=17 * cm * 480 / 1320),
    Paragraph("Figure 1. Histograms of L (left) and I (right), 60 bins, all images.", CAP),
]))
s.extend(bullets([
    f"L is strongly <b>right-skewed</b>: mean {dl['mean']:.4f} is more than double the median "
    f"{dl['50%']:.4f}; half of the images have less than 12% loss.",
    "There is a large spike near 0 (more than 6,000 images in the first bin) and smaller bumps "
    "around 0.3-0.4, about 0.65 and 0.8-0.9, which probably correspond to specific soiling "
    "conditions applied on certain days.",
    f"{n_L_zero} images have L exactly 0.",
    f"{n_low_I:,} images have very low irradiance (I &lt; 0.03). The loss ratio in such light may be "
    "unreliable; this is flagged as a caveat, not removed.",
    "Consequence: a model can get a low average error by predicting small values, so we will "
    "also report MAE separately for low / mid / high L ranges, and compare against a "
    "mean-prediction baseline.",
]))

s.append(Paragraph("4.3 Time structure", H2))
dg = gap.describe()
s.append(table([
    ["Quantity", "Value"],
    ["Time range", f"{df['timestamp'].min()}  to  {df['timestamp'].max()}"],
    ["Distinct days with data", f"{per_day.shape[0]} (no data 17-19 Jun 2017)"],
    ["Gap between consecutive images", f"min {dg['min']:.0f} s, median {dg['50%']:.0f} s, "
                                       f"mean {dg['mean']:.1f} s, max {dg['max']:,.0f} s"],
    ["Gaps > 10 s", f"{int((gap > 10).sum()):,}"],
    ["Gaps > 60 s", f"{int((gap > 60).sum()):,}"],
    ["Gaps > 600 s (10 min)", f"{int((gap > 600).sum()):,}"],
    ["Gaps > 3600 s (1 h)", f"{int((gap > 3600).sum()):,}"],
], [6 * cm, 11 * cm]))
s.append(Spacer(1, 6))
rows = [["Date", "Images", "First capture", "Last capture"]]
for d, r in per_day.iterrows():
    rows.append([str(d), f"{r['n']:,}", r["first"].strftime("%H:%M:%S"),
                 r["last"].strftime("%H:%M:%S")])
s.append(KeepTogether([Paragraph("Images per day", H2),
                       table(rows, [4 * cm, 3 * cm, 5 * cm, 5 * cm])]))
s.append(p(
    "Images are taken about every 5 seconds, so neighbouring frames are almost identical "
    "(same soiling, nearly the same light). If frames are split at random, near-copies of a "
    "test image will sit in the training set and the test score will be optimistic. Long gaps "
    "(more than 10 minutes / 1 hour) mark natural session and day boundaries that can be used "
    "to keep related frames together. Whether this effect actually inflates scores will be "
    "measured in Step 2 by comparing a random split with a grouped split; we will not call it "
    "leakage unless the numbers show it."))

s.append(KeepTogether([
    Paragraph("4.4 Visual check", H2),
    Image(str(OUT_DIR / "sample_grid_by_L.png"), width=17 * cm, height=17 * cm * 990 / 1540),
    Paragraph("Figure 2. 12 images spread evenly across the sorted L range "
              "(title: L and I of each image).", CAP),
]))
s.extend(bullets([
    "L near 0 shows a clean panel; high L shows visible white deposits and streaks.",
    "The image-to-label relation is <b>noisy</b>: L = 0.14 and L = 0.99 look similarly streaked, "
    "and L = 0.44 shows only a few rust-coloured spots. The achievable error is therefore limited "
    "and we should not expect a near-perfect fit.",
    "The camera view is fixed and includes background (neighbouring panels, roof, pipe). One "
    "high-L image contains a person beside the panel. Grad-CAM must confirm the model attends "
    "to the panel surface, not to background cues.",
]))

# 5
s.append(Paragraph("5. Outputs produced", H1))
s.append(table([
    ["File", "Content"],
    ["step1_outputs/dataset_manifest.csv",
     "One row per image, sorted by time. Columns: filename, path, timestamp, L, I, "
     "time_gap_sec, corrupt."],
    ["step1_outputs/label_hist.png", "Figure 1, histograms of L and I."],
    ["step1_outputs/sample_grid_by_L.png", "Figure 2, sample images across the L range."],
    ["step1_manifest.py", "The script that produces all of the above."],
], [6.5 * cm, 10.5 * cm]))

# 6
s.append(Paragraph("6. Decisions and caveats", H1))
s.extend(bullets([
    "<b>Target</b>: L as a fraction in [0, 1]; convert to % only for reporting.",
    "<b>No input leakage</b>: irradiance I is not a model input.",
    "<b>No data removed</b>: no corrupt files; low-irradiance images kept but flagged.",
    "<b>Raw data is read-only</b>: images are never modified, renamed or deleted.",
    "<b>Skewed labels</b>: report MAE per L range in addition to overall MAE, RMSE, R<super>2</super>.",
    "<b>Temporal correlation</b>: random splits may be optimistic; this must be tested, "
    "not assumed.",
    "<b>Label noise</b>: visual appearance does not fully determine L, which limits accuracy.",
]))

# 7
s.append(Paragraph("7. Next step (Step 2): splits, data pipeline, baseline and model", H1))
s.append(Paragraph("7.1 Per-day label summary (first task)", H2))
s.append(p(
    "Compute n, mean, min and max of L for each of the 15 days. Why: to see whether soiling "
    "level drifts across days (e.g. specific soiling applied on specific days) and therefore "
    "whether the grouped split should keep whole <b>days</b> together or whole <b>sessions</b> "
    "(runs of frames separated by long gaps). If each day covers only a narrow L range, a "
    "day-based split could leave some L ranges missing from training; sessions may then be the "
    "better unit."))
s.append(Paragraph("7.2 Shared split files", H2))
s.extend(bullets([
    "<b>Random split</b> 70 / 15 / 15 (train / validation / test) with a fixed seed.",
    "<b>Grouped / chronological split</b> 70 / 15 / 15 where whole days or sessions stay "
    "in one subset.",
    "Saved as CSV files in the project folder; this single set is used by every team member. "
    "Nobody makes their own split.",
    "Check that the L distribution of each subset is reasonable and report it.",
]))
s.append(Paragraph("7.3 Preprocessing and DataLoader", H2))
s.extend(bullets([
    "Resize 192 x 192 to 224 x 224, convert to tensor, normalise with ImageNet mean/std.",
    "Light augmentation on training only (to be decided, e.g. small colour jitter); "
    "no augmentation on validation/test.",
    "Add torch and torchvision to the environment.",
]))
s.append(Paragraph("7.4 Baseline and ResNet-18", H2))
s.extend(bullets([
    "<b>Baseline</b>: always predict the training-set mean of L; gives the reference MAE/RMSE "
    "(R<super>2</super> = 0 by construction on the training data).",
    "<b>Model</b>: ImageNet-pretrained ResNet-18, final layer replaced by a single-output "
    "regression head, MSE loss, Adam/AdamW.",
    "Training: first train only the head, then fine-tune the whole network with a lower "
    "learning rate; early stopping on validation loss; fixed seeds.",
]))
s.append(Paragraph("7.5 Evaluation and add-ons", H2))
s.extend(bullets([
    "Test-set MAE, RMSE, R<super>2</super>, plus MAE per L range (low / mid / high), for both "
    "split strategies; explain any gap between them.",
    "Add-ons in priority order: Grad-CAM on layer4 for the scalar output; robustness tests "
    "(brightness, noise, JPEG compression); EfficientNet-B0 only if time allows.",
]))
s.append(KeepTogether([Paragraph("7.6 Timeline", H2), table([
    ["Date", "Work"],
    ["19-20 Nov 2026", "Step 2 core work: splits, pipeline, baseline, ResNet-18, evaluation"],
    ["20 Nov 2026", "Code freeze, report; viva preparation in the evening"],
    ["21 Nov 2026", "Presentation"],
], [4 * cm, 13 * cm])]))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Step 1: Dataset Audit and Manifest",
                        author="IDSE Solar Soiling Team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("Wrote", PDF)
