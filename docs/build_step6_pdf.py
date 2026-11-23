"""Build docs/Step6_GradCAM.pdf (Step 6 documentation).

Usage (from the project folder, venv active):
    python docs/build_step6_pdf.py

Nothing is retrained or re-run here. The document uses the files written by gradcam.py
(outputs/gradcam_grouped/*.png and examples_used.csv), the Step 4 prediction file
(outputs/preds/resnet18_grouped_test.csv) and splits/splits.csv. Small extra facts (the size of
the candidate pools per category, day/time/irradiance of the examples, the difference between the
Grad-CAM forward pass and the saved prediction) are computed here from those saved files only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
GC = OUT / "gradcam_grouped"
PDF = ROOT / "docs" / "Step6_GradCAM.pdf"

# ---------------------------------------------------------------- load
sp = pd.read_csv(ROOT / "splits" / "splits.csv", parse_dates=["timestamp"])
ex = pd.read_csv(GC / "examples_used.csv")
pr = pd.read_csv(OUT / "preds" / "resnet18_grouped_test.csv")
ex = ex.merge(pr.rename(columns={"L_pred": "L_pred_saved", "L_true": "L_true_saved"}),
              on="filename", how="left") \
       .merge(sp[["filename", "timestamp", "day", "I"]], on="filename", how="left")
merge_ok = ex.L_pred_saved.notna().all() and ex.day.notna().all()
ex["d_pred"] = (ex.L_pred - ex.L_pred_saved).abs()
max_dpred = ex.d_pred.max()

# candidate pools, exactly as in gradcam.py
pr["err"] = (pr.L_pred - pr.L_true).abs()
pr = pr.merge(sp[["filename", "day", "timestamp", "I"]], on="filename", how="left")
good = pr[pr.err < 0.05]
pools = {
    "clean_correct": good[good.L_true < 0.05],
    "moderate_correct": good[(good.L_true >= 0.25) & (good.L_true <= 0.45)],
    "heavy_correct": good[good.L_true >= 0.6],
}
pool_info = {k: (len(v), v.day.value_counts().to_dict()) for k, v in pools.items()}
n_good = len(good)
# worst-failure selection rebuilt and compared with the CSV
wf = pr.assign(k=pr.L_true.round(3)).sort_values("err", ascending=False).drop_duplicates("k").head(4)
wf_ok = list(wf.filename) == list(ex[ex.category == "worst_failures"].filename)
# reproduce: the same 4 examples per category with seed 0
rep_ok = all(list(v.sample(min(4, len(v)), random_state=0).filename) ==
             list(ex[ex.category == k].filename) for k, v in pools.items())

# ---------------------------------------------------------------- styles (same as Steps 1-5)
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
                      "IDSE project - Solar panel soiling power-loss regression - Step 6 documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def day_str(d):
    return ", ".join(f"{k[-2:]} Jun: {v:,}" for k, v in sorted(d.items()))


# ---------------------------------------------------------------- content
s = []
s.append(Paragraph("Step 6: Grad-CAM Interpretability", ss["Title"]))
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
                   "Status: Step 6 <b>done</b>; its hypotheses were tested quantitatively in Step 7 "
                   "(robustness and occlusion, separate document).", CAP))

# summary
s.append(Paragraph("Summary of results", H1))
s.extend(bullets([
    "<b>What:</b> Grad-CAM heatmaps for the grouped-split ResNet-18 (the model behind the headline "
    "result, grouped test MAE 0.0869, " + R2 + " 0.706), computed on the last residual stage "
    "(" + C % "layer4" + ", 7x7 feature maps) with the <b>scalar predicted L</b> as the target, for 16 "
    "grouped test images: 4 clean, 4 moderate and 4 heavy correct predictions (|error| &lt; 0.05) and "
    "the 4 worst failures.",
    "<b>Heavy soiling (true 0.63-0.66, predicted 0.63-0.69):</b> the hot region lies over the white "
    "deposits on the panel, with additional strong heat along the panel's right frame edge. This "
    "<b>supports</b> the use of real soiling evidence.",
    "<b>Clean panels (true 0.03-0.05):</b> the panel surface is cold; the heat sits on the stand/base "
    "below the panel and on the top-right corner where a neighbouring panel appears.",
    "<b>Moderate (true 0.31-0.34):</b> the two small rust-coloured spots near the top of the panel are "
    "<b>not</b> highlighted; heat is again on the stand and the top-right corner plus part of the "
    "lower panel.",
    "<b>Worst failures (true 0.86-0.99, predicted 0.04-0.09):</b> the frames look clean and the "
    "heatmaps look the same as for clean panels: nothing on the panel surface. This is consistent "
    "with the Step 5 hypothesis that these labels do not match the visible panel state.",
    "<b>Conclusion (report wording):</b> Grad-CAM <i>partly supports attention to soiled regions, but "
    "also shows reliance on regions outside the soiled area</i>. Because Grad-CAM is coarse and not "
    "causal, the stand and top-right hot spots were turned into hypotheses and tested with occlusion "
    "in Step 7 (result there: the stand is not needed, the panel interior and the top-right corner "
    "are).",
]))
s.append(Paragraph(f"Checks: examples reproduced from the saved predictions with the same rules and seed: "
                   f"{ok(rep_ok and wf_ok)}; every example found in the prediction and split files: "
                   f"{ok(merge_ok)}.", CAP))

# 1 purpose
s.append(Paragraph("1. Purpose of Step 6 and where it fits", H1))
s.append(p(
    "Steps 4 and 5 showed <i>how well</i> the model predicts the power loss L. They do not show "
    "<i>what in the image</i> the prediction is based on. A model can reach a good score for the wrong "
    "reason, for example by recognising the time of day, the lighting, a person standing next to the "
    "panel (seen in one high-L image in Step 1) or another object in the fixed camera view that happens "
    "to correlate with L in the 10 training days. Such shortcuts would not transfer to another site "
    "or day. Step 6 therefore asks:"))
s.extend(bullets([
    "Does the network look at the <b>panel surface</b> and the <b>soiling</b> on it, or at the "
    "background (ground, stand, neighbouring panels)?",
    "Does the evidence differ between clean, moderately and heavily soiled panels in a sensible way?",
    "What does the model 'see' in the worst failures from Step 5 (frames labelled with almost total "
    "loss but predicted almost clean)?",
]))
s.append(p(
    "This was planned as the first add-on after the core results (Grad-CAM on the scalar "
    "regression output, " + C % "layer4" + "), and it corresponds to the interpretability part of the "
    "team's literature survey (Selvaraju et al., Grad-CAM, ICCV 2017). No model is trained or changed "
    "in this step; raw images are only read."))

# 2 what & why
s.append(Paragraph("2. What we did and why", H1))
s.append(table([
    ["#", "Action", "Why"],
    ["6.1", "Wrote " + C % "gradcam.py" + ": loads the best grouped checkpoint, hooks " +
     C % "layer4" + ", computes a Grad-CAM heatmap for one image at a time.",
     "Grad-CAM is the standard, cheap, model-agnostic way to see which image regions drive a CNN "
     "output; it needs no retraining and works on the trained model as it is."],
    ["6.2", "Used the <b>scalar predicted L</b> as the quantity whose gradient is taken (instead of "
     "a class score).",
     "Our model is a regression model with one output. The gradient of that output shows which "
     "regions would raise the predicted power loss."],
    ["6.3", "Used the <b>grouped</b> checkpoint and <b>grouped test</b> images.",
     "The grouped model gives the honest headline result; its test days were never used for "
     "training or model selection, so the maps show behaviour on unseen days."],
    ["6.4", "Picked 4 examples in each of 3 correct categories (clean, moderate, heavy) and the 4 "
     "worst failures (one per distinct label value).",
     "Correct cases show what the model uses when it works; comparing categories shows whether the "
     "evidence changes with soiling; failures show what goes wrong. One per label because "
     "consecutive frames share the same label (Step 5), so the top errors would otherwise be "
     "near-identical frames."],
    ["6.5", "Saved one figure per category (original image on top, overlay below) plus " +
     C % "examples_used.csv" + ".",
     "Report/slide figures; the CSV makes the selection traceable."],
    ["6.6", "Inspected the maps and wrote down the patterns; turned the background hot spots into "
     "hypotheses for Step 7.",
     "A heatmap is not proof; any claim about reliance on a region must be tested by changing that "
     "region and measuring the error (done in Step 7)."],
], [0.9 * cm, 7.6 * cm, 8.5 * cm], small=True))

# 3 method
s.append(Paragraph("3. Method", H1))
s.append(Paragraph("3.1 Command", H2))
s.append(Preformatted(".\\.venv\\Scripts\\Activate.ps1\npython gradcam.py                  "
                      "# grouped checkpoint, 4 examples per category\n"
                      "python gradcam.py --strategy random  # optional, not used in this report",
                      CODE))
s.append(Paragraph("3.2 Grad-CAM for a regression output", H2))
s.append(p(
    "Let A<super>k</super> (k = 1 ... 512) be the 512 feature maps produced by " + C % "layer4" +
    " for one image (each 7x7, because ResNet-18 reduces a 224x224 input by a factor of 32), and let "
    "y be the network's raw output, the predicted L. Grad-CAM computes:"))
s.append(table([
    ["Step", "Formula", "Meaning"],
    ["1. Forward", "y = f(x)", "Predicted L for image x (one number)."],
    ["2. Backward", "dy / dA<super>k</super><sub>ij</sub>",
     "How much y would change if feature map k at position (i, j) grew."],
    ["3. Channel weights", "w<sub>k</sub> = (1/49) sum<sub>i,j</sub> dy / dA<super>k</super><sub>ij</sub>",
     "Importance of channel k for raising the prediction (global average of its gradient)."],
    ["4. Map", "M = ReLU( sum<sub>k</sub> w<sub>k</sub> A<super>k</super> )",
     "7x7 map; ReLU keeps only regions whose features push y <b>up</b>."],
    ["5. Display", "bilinear upsample 7x7 -&gt; 224x224, min-max scale to [0, 1], jet colour "
     "map, alpha 0.45 over the image", "Red = strongest positive evidence in this image, blue = none."],
], [2.8 * cm, 6.4 * cm, 7.8 * cm], small=True))
s.append(p(
    "In the original paper y is the score of one class. With a single regression output the same "
    "formula applies directly, and the interpretation becomes: <b>red regions are those whose "
    "features increase the predicted power loss</b>. Regions that lower the prediction (for example a "
    "clean, dark panel surface) are removed by the ReLU and appear blue."))
s.append(Paragraph("3.3 Why layer4", H2))
s.extend(bullets([
    "It is the last convolutional stage before global average pooling and the linear head, so its "
    "features are the most task-specific (they directly feed the prediction) while still having a "
    "spatial layout. This is the layer recommended by the Grad-CAM authors.",
    "Trade-off: the map is only 7x7, so each cell covers about 32x32 pixels of the 224x224 input "
    "(about 27x27 pixels of the original 192x192 frame). Small features such as the rust spots "
    "occupy roughly one cell and can be lost. Earlier layers (" + C % "layer3" + ", 14x14) would be "
    "sharper but less directly tied to the output; they were not used.",
]))
s.append(Paragraph("3.4 Implementation details (" + C % "gradcam.py" + ")", H2))
s.extend(bullets([
    "Model: " + C % "build_model(pretrained=False)" + " (ResNet-18 with " + C % "fc" + " replaced by "
    "one linear output, same as " + C % "train.py" + "), weights from " +
    C % "outputs/checkpoints/resnet18_grouped_best.pt" + ", set to <b>eval</b> mode (BatchNorm uses "
    "its running statistics; there is no dropout in ResNet-18), on the GPU.",
    "Hooks: a forward hook on " + C % "model.layer4" + " stores the activations; a full backward hook "
    "stores the gradient of the output with respect to them. " + C % "model.zero_grad()" + " is "
    "called before each image so gradients do not accumulate.",
    "Input: the val/test transform from " + C % "data.py" + " (resize 192 -&gt; 224, tensor, ImageNet "
    "normalisation, <b>no augmentation</b>), one image at a time (batch of 1).",
    "Precision: the Grad-CAM forward pass runs in full fp32 (no bf16 autocast, unlike the Step 4 "
    f"evaluation). The predictions shown in the figure titles therefore differ very slightly from the "
    f"saved Step 4 predictions: maximum absolute difference over the 16 examples = "
    f"<b>{max_dpred:.4f}</b>.",
    "The displayed prediction is clamped to [0, 1], as in evaluation; the gradient is taken on the "
    "unclamped output (clamping would make the gradient zero outside [0, 1]).",
    "Display: the normalised tensor is converted back to RGB (multiply by std, add mean) for the top "
    "row; the bottom row overlays the heatmap with the jet colour map and alpha 0.45.",
    "Each heatmap is min-max scaled <b>within its own image</b>. Colours show where the evidence is "
    "strongest relative to the rest of that image; they cannot be compared in absolute strength "
    "between images (a red spot on a clean panel is not 'as much evidence' as a red patch on a "
    "heavily soiled panel).",
]))
s.append(Paragraph("3.5 Example selection", H2))
s.append(p(
    "All examples come from " + C % "outputs/preds/resnet18_grouped_test.csv" + " (6,733 grouped test "
    f"frames, days 21 and 29 Jun). 'Correct' means |predicted - true| &lt; 0.05 ({n_good:,} frames "
    "qualify). Within each pool, 4 frames were drawn at random with seed 0 "
    "(" + C % "DataFrame.sample(random_state=0)" + ")."))
s.append(table([
    ["Category", "Rule", "Pool size", "Pool by day"],
    ["clean_correct", "|error| &lt; 0.05 and true L &lt; 0.05", f"{pool_info['clean_correct'][0]:,}",
     day_str(pool_info["clean_correct"][1])],
    ["moderate_correct", "|error| &lt; 0.05 and 0.25 &lt;= true L &lt;= 0.45",
     f"{pool_info['moderate_correct'][0]:,}", day_str(pool_info["moderate_correct"][1])],
    ["heavy_correct", "|error| &lt; 0.05 and true L &gt;= 0.6", f"{pool_info['heavy_correct'][0]:,}",
     day_str(pool_info["heavy_correct"][1])],
    ["worst_failures", "largest |error|, after keeping only the first frame of each distinct true L "
     "(rounded to 3 decimals); top 4", "-", "-"],
], [3 * cm, 7 * cm, 2 * cm, 5 * cm], small=True))
s.append(Paragraph("Pool sizes computed here from the saved predictions with the same rules as "
                   "gradcam.py.", CAP))
s.append(p(
    "The moderate band (0.25-0.45) leaves a gap to the clean band (&lt; 0.05) and the heavy band "
    "(&gt;= 0.6) so that the three categories are visually distinct. Because the grouped test set "
    "has only two days, the categories are tied to days: the 12 correct examples that were drawn come "
    "from 21 Jun (clean, moderate) and 29 Jun (heavy). The heavy examples therefore all show the "
    "same deposit pattern of 29 Jun before the label drop at about 12:25 that Step 5 interpreted as "
    "a cleaning (Section 5)."))

# 4 examples table
s.append(Paragraph("4. The 16 examples used", H1))
rows = [["Category", "Day / time", "I", "True L", "Pred. (Grad-CAM run)", "Pred. (saved, Step 4)"]]
for r in ex.itertuples():
    rows.append([r.category, f"{r.day[-2:]} Jun {r.timestamp:%H:%M:%S}", f"{r.I:.3f}",
                 f"{r.L_true:.4f}", f"{r.L_pred:.4f}", f"{r.L_pred_saved:.4f}"])
s.append(table(rows, [3.2 * cm, 3.2 * cm, 1.6 * cm, 2 * cm, 3.5 * cm, 3.5 * cm], small=True,
               shade_rows={i: "#FCE8E6" for i in range(13, 17)}))
s.append(Paragraph("Source: outputs/gradcam_grouped/examples_used.csv joined with splits/splits.csv "
                   "and outputs/preds/resnet18_grouped_test.csv. Red rows: worst failures.", CAP))
s.extend(bullets([
    "Two moderate examples and one heavy example were taken in low light (I = 0.061, 0.163 and "
    "0.036); they are still predicted correctly, so poor light alone does not break the model.",
    "The 4 worst failures are 29 Jun 13:49 and 13:31 and 21 Jun 17:49 and 17:58 - four of the five "
    "time windows identified in Step 5. The 29 Jun 12:27 window and the 21 Jun 13:15 frame are not in "
    "this figure because only the top 4 distinct labels were drawn.",
]))

# 5 results by category
s.append(Paragraph("5. Results by category", H1))
s.append(p("Each figure: top row = the input image as the network sees it (224x224, de-normalised), "
           "with true and predicted L; bottom row = Grad-CAM overlay (red = strongest evidence for a "
           "higher predicted L in that image)."))

s.append(Paragraph("5.1 Heavy soiling, correct (true 0.63-0.66, predicted 0.63-0.69)", H2))
s.append(fig(GC / "heavy_correct.png", 17,
             "Figure 1. Grad-CAM, heavy soiling, correct predictions (outputs/gradcam_grouped/"
             "heavy_correct.png). All four frames are from 29 Jun."))
s.extend(bullets([
    "Large white deposits cover the centre-right of the panel. The hot region (yellow/orange) lies "
    "<b>over these deposits</b> and follows their shape roughly at the 7x7 resolution.",
    "There is additional strong heat (red) along the panel's <b>right frame edge</b>, next to the "
    "rightmost deposits.",
    "The clean left part of the panel and the background above the panel are blue.",
    "The stand area below the panel is only weakly warm (green/yellow), much weaker than the deposits.",
    "<b>Reading:</b> for heavy soiling the model's evidence for a high L is located on the soiled "
    "panel area. This is the behaviour we want and supports the claim that the model uses real "
    "soiling evidence.",
]))

s.append(Paragraph("5.2 Clean panels, correct (true 0.03-0.05, predicted 0.04-0.07)", H2))
s.append(fig(GC / "clean_correct.png", 17,
             "Figure 2. Grad-CAM, clean panels, correct predictions (outputs/gradcam_grouped/"
             "clean_correct.png). All four frames are from 21 Jun."))
s.extend(bullets([
    "The panel surface is uniformly blue: no part of the panel pushes the prediction up.",
    "The strongest heat is on the <b>base/stand below the panel</b> (red at the bottom, around the "
    "gravel and the support) and on the <b>top-right corner</b>, where the panel's corner meets a "
    "neighbouring panel.",
    "<b>Reading:</b> because each map is scaled within its own image and the predicted L is close to "
    "zero, these red spots are the <i>relatively</i> strongest positive evidence in an image that has "
    "very little. They do not necessarily mean that the stand drives the prediction strongly, but they "
    "do show that whatever small positive evidence exists is located outside the panel surface. This "
    "is the observation that motivated the occlusion test in Step 7.",
]))

s.append(Paragraph("5.3 Moderate soiling, correct (true 0.31-0.34, predicted 0.31-0.39)", H2))
s.append(fig(GC / "moderate_correct.png", 17,
             "Figure 3. Grad-CAM, moderate soiling, correct predictions (outputs/gradcam_grouped/"
             "moderate_correct.png). All four frames are from 21 Jun morning (07:07-08:52)."))
s.extend(bullets([
    "The visible soiling is two small rust-coloured spots near the top-centre of the panel. They are "
    "<b>not</b> highlighted: the cells containing them are blue.",
    "Heat appears again on the stand (red at the bottom) and the top-right corner, plus a broader "
    "green/yellow area over the lower-left/central part of the panel (strongest in the second "
    "example, predicted 0.39).",
    "<b>Reading:</b> the model predicts the moderate L correctly but the map does not show it using "
    "the rust spots. Possible explanations: (a) the spots are about one 7x7 cell in size and are "
    "averaged away at this resolution; (b) the model uses diffuse cues spread over the lower panel "
    "(e.g. a thin dust film or a change in reflectance) that are not visible as distinct spots; "
    "(c) the prediction partly relies on regions outside the soiled area. Grad-CAM alone cannot "
    "decide between these.",
    "Note that in the moderate images the true L (about 0.33) is much higher than the few spots "
    "would suggest; Step 1 already found that the image-to-label relation is noisy (e.g. images with "
    "only rust spots can carry high L).",
]))

s.append(Paragraph("5.4 Worst failures (true 0.86-0.99, predicted 0.04-0.09)", H2))
s.append(fig(GC / "worst_failures.png", 17,
             "Figure 4. Grad-CAM, the four worst grouped test predictions, one per distinct label "
             "(outputs/gradcam_grouped/worst_failures.png): 29 Jun 13:49, 29 Jun 13:31, "
             "21 Jun 17:49, 21 Jun 17:58."))
s.extend(bullets([
    "The four images show a <b>clean-looking panel</b>: no deposits, no streaks, no visible shading.",
    "The heatmaps look <b>the same as for the clean, correctly predicted panels</b> (Figure 2): "
    "cold panel surface, heat on the stand and the top-right corner.",
    "<b>Reading:</b> the model treats these frames exactly like clean panels, which is what they look "
    "like. The network is not attending to something wrong; there is simply nothing on the panel "
    "that would explain a near-total power loss. This is consistent with the Step 5 hypothesis that "
    "these labels (or the measurement at those moments) do not match the visible panel state. The "
    "21 Jun 17:49 and 17:58 frames also have very low irradiance (I = 0.052 and 0.027), where a loss "
    "ratio is numerically unreliable.",
    "Caution: this is a consistency argument, not proof that the labels are wrong. A cause that is "
    "invisible in an RGB image (an electrical fault, partial shading of a reference cell, a logging "
    "delay) would produce the same picture.",
]))

# 6 cross-category
s.append(Paragraph("6. Patterns across all categories", H1))
s.append(table([
    ["Region", "Clean", "Moderate", "Heavy", "Worst failures"],
    ["Panel surface (clean part)", "cold", "partly warm (lower panel)", "cold", "cold"],
    ["Soiled area on the panel", "-", "rust spots not highlighted", "<b>hot</b> (deposits)", "-"],
    ["Right frame edge", "weak", "weak", "<b>hot</b>", "weak"],
    ["Stand / base below panel", "<b>hot</b>", "<b>hot</b>", "weakly warm", "<b>hot</b>"],
    ["Top-right corner (neighbouring panel)", "<b>hot</b>", "hot", "warm", "<b>hot</b>"],
    ["Background above/left of panel", "cold", "cold", "cold", "cold"],
], [5 * cm, 2.5 * cm, 3.6 * cm, 2.6 * cm, 3.3 * cm], small=True))
s.append(Paragraph("Qualitative summary from visual inspection of Figures 1-4 (relative to each "
                   "image's own maximum).", CAP))
s.extend(bullets([
    "The stand-base and top-right hot spots appear in <b>almost every image regardless of class</b>. "
    "A region that is highlighted whatever the soiling level is suspicious: it could be a shortcut "
    "(something in the fixed camera view that correlates with L in the training days), or it could "
    "be a constant offset that the network adds to every prediction and that Grad-CAM shows simply "
    "because there is nothing stronger in clean images.",
    "Only for heavy soiling does the strongest evidence move onto the panel surface. Since the "
    "camera position is fixed for the whole dataset, a background cue cannot be ruled out from the "
    "maps alone.",
    "No heat was seen on people or other transient objects in these 16 frames (the Step 1 frame with "
    "a person next to the panel is not among them, so it was not checked specifically).",
]))

# 7 interpretation & link to step 7
s.append(Paragraph("7. Interpretation and the hypotheses passed to Step 7", H1))
s.append(p(
    "<b>Report wording:</b> \"Grad-CAM partly supports attention to soiled regions (heavy deposits "
    "are highlighted), but also shows reliance on regions outside the soiled area (the stand below the "
    "panel and the top-right corner near a neighbouring panel).\""))
s.append(p("Because Grad-CAM shows correlation with the output and not causation, the observations "
           "were turned into testable hypotheses for Step 7:"))
s.append(table([
    ["Hypothesis from Grad-CAM", "Test in Step 7", "Step 7 result (grouped test, n = 6,733)"],
    ["H1: the model depends on the stand/base region.",
     "Grey out the bottom 20% of the image (rows 179-223).",
     "<b>Rejected:</b> MAE 0.0791 vs 0.0869 clean (slightly better), mean prediction shift 0.032."],
    ["H2: the model depends on the top-right corner.",
     "Grey out rows 0-55, cols 168-223.",
     "<b>Supported:</b> MAE 0.1255 (+44%), bias +0.076; but the box also covers the panel's own "
     "top-right corner (cause not established)."],
    ["H3: predictions depend mainly on the panel surface.",
     "Grey out a 56x56 box at the panel centre (rows/cols 84-139).",
     "<b>Supported:</b> MAE 0.1725 (about 2x), " + R2 + " 0.224."],
    ["Control: a grey box alone does not cause errors.",
     "Same-size boxes in background corners.",
     "Top-left MAE 0.0869, bottom-left 0.0844: no effect."],
], [4.6 * cm, 5 * cm, 7.4 * cm], small=True))
s.append(Paragraph("Numbers from outputs/eval/robustness_grouped_test.csv and "
                   "robustness_grouped_test_controls.csv; details in the Step 7 document.", CAP))
s.append(p(
    "Combined conclusion of Steps 6 and 7: the stand hot spot seen by Grad-CAM is <b>not</b> a real "
    "dependence (removing it does not hurt), the predictions depend mainly on the panel area, and "
    "the top-right corner region also matters for reasons that are not yet clear. Proposed future "
    "work (also in the survey): crop or standardise the panel so it fills the frame, which removes "
    "the stand and the neighbouring panel from the input."))

# 8 limitations
s.append(Paragraph("8. Limitations of Grad-CAM in this project", H1))
s.extend(bullets([
    "<b>Coarse resolution:</b> 7x7 cells of about 32x32 input pixels; small rust spots cover about one "
    "cell and can be invisible; bilinear upsampling makes the map look smoother than it is.",
    "<b>ReLU keeps only positive evidence:</b> regions that <i>lower</i> the predicted loss (e.g. a "
    "clean dark panel) are not shown at all. For clean panels, where the prediction is near zero, the "
    "interesting evidence is mostly negative and therefore invisible.",
    "<b>Per-image normalisation:</b> colours are relative within an image; a red stand on a clean "
    "panel and red deposits on a dirty panel are not the same amount of evidence.",
    "<b>Not causal:</b> the map shows where gradients and activations are large, not what would happen "
    "if a region changed. Causal claims require interventions (Step 7 occlusion).",
    "<b>Small, non-random sample:</b> 16 images, 4 per category, from 2 test days; correct categories "
    "were drawn at random within strict pools but each category comes from one day. The observations "
    "are qualitative and from visual inspection only; no quantitative heatmap statistic was computed.",
    "<b>Single model and layer:</b> only the grouped checkpoint and " + C % "layer4" + " were "
    "examined; the random-split model and other layers or methods (e.g. Grad-CAM++, integrated "
    "gradients) were not compared.",
]))

# 9 outputs
s.append(Paragraph("9. Outputs produced", H1))
s.append(table([
    ["File", "Content"],
    ["gradcam.py", "Step 6 script (GradCAM class with layer4 hooks; example selection; figures)."],
    ["outputs/gradcam_grouped/clean_correct.png", "Figure 2: 4 clean correct examples + overlays."],
    ["outputs/gradcam_grouped/moderate_correct.png", "Figure 3: 4 moderate correct examples + overlays."],
    ["outputs/gradcam_grouped/heavy_correct.png", "Figure 1: 4 heavy correct examples + overlays."],
    ["outputs/gradcam_grouped/worst_failures.png", "Figure 4: 4 worst failures + overlays."],
    ["outputs/gradcam_grouped/examples_used.csv",
     "category, filename, L_true, L_pred (Grad-CAM fp32 run) for all 16 examples."],
    ["docs/build_step6_pdf.py", "Builds this document from the files above (no model run)."],
], [7.4 * cm, 9.6 * cm], small=True))

# 10 decisions & caveats
s.append(Paragraph("10. Decisions and caveats", H1))
s.extend(bullets([
    "<b>Grouped model only</b>, because it is the headline model; the random-split model was not "
    "visualised for the report (the script supports it with " + C % "--strategy random" + ").",
    "<b>Test images, not training images</b>, so the maps show behaviour on unseen days. Using the "
    "test set for visualisation does not change any model or number.",
    "<b>Fixed seed (0)</b> for example selection, so the same figures are produced on every run "
    "(verified here: " + ok(rep_ok) + ").",
    "<b>Failures one per distinct label value</b>, because consecutive frames share labels and would "
    "otherwise fill the figure with near-duplicates.",
    "<b>Gradient on the unclamped output</b>; predictions in titles are clamped to [0, 1]. Grad-CAM "
    "runs in fp32 while Step 4 evaluated in bf16 (difference at most " + f"{max_dpred:.4f}" + ").",
    "<b>Observations are visual and qualitative</b>; anything stated as a dependence was tested in "
    "Step 7 before being used in the report.",
]))

# 11 viva
s.append(KeepTogether([Paragraph("11. Viva points from this step", H1), table([
    ["Question", "Answer grounded in Step 6"],
    ["What is Grad-CAM?",
     "Gradient-weighted class activation mapping: weights the last convolutional feature maps by "
     "the average gradient of the output and keeps the positive part. It shows which regions push the "
     "output up."],
    ["How do you use it for regression?",
     "The target is the single predicted L instead of a class score; red regions are those that "
     "increase the predicted power loss."],
    ["Why layer4?", "Last convolutional stage, most task-specific features, still spatial (7x7). "
     "Cost: coarse maps."],
    ["Does the model look at the dirt?",
     "For heavy soiling yes, the deposits are highlighted. For clean and moderate panels the "
     "strongest evidence is on the stand and the top-right corner; Step 7 showed the stand is not "
     "needed, while the panel interior is."],
    ["Why are the rust spots not highlighted?",
     "They cover about one 7x7 cell; ReLU and normalisation hide weak evidence; the model may use "
     "diffuse cues. Grad-CAM is coarse."],
    ["What do the worst failures show?",
     "Clean-looking panels with clean-panel heatmaps: the model sees a clean panel, the label says "
     "~0.9-0.99 loss. Consistent with label/measurement anomalies (hypothesis)."],
    ["Is Grad-CAM proof?", "No. It shows where the model looked, not causation; that is why we "
     "ran the occlusion tests."],
], [5 * cm, 12 * cm], small=True)]))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Step 6: Grad-CAM Interpretability",
                        author="IDSE Solar Soiling Team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("merge ok:", merge_ok, "| selection reproduced:", rep_ok, "| worst reproduced:", wf_ok)
print("max |gradcam pred - saved pred|:", round(float(max_dpred), 5))
print("pools:", pool_info, "| n good:", n_good)
print(ex[["category", "day", "timestamp", "I", "L_true", "L_pred", "L_pred_saved"]].to_string())
print("Wrote", PDF)
