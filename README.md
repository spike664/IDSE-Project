# Image-based estimation of solar panel soiling power loss

IDSE course project (IIITDM Kancheepuram), Team 8: Mahalis Rayhan (CS24B1095), Manieesh Manohar (CS24B1005),
Siddharth (CS24B1013), Ram Niwas (CS24I1001).

Predicts the continuous soiling-induced power loss `L` (a fraction in [0, 1]) from a single RGB image of a
solar panel, using the DeepSolarEye dataset and an ImageNet-pretrained CNN (ResNet-18) with a one-output
regression head. Irradiance is never used as an input.

## Main results (day-based split, test = 21 and 29 June)

| Model | Test MAE | Test R² |
|---|---|---|
| Mean predictor (baseline) | 0.2100 | -0.003 |
| ResNet-18 (seed 42) | 0.0869 | 0.706 |
| ResNet-18 (mean of 3 seeds) | 0.0877 ± 0.0018 | 0.702 ± 0.010 |
| EfficientNet-B0 (mean of 3 seeds) | 0.0885 ± 0.0035 | 0.681 ± 0.022 |

A random split gives MAE 0.047 (R² 0.913) for ResNet-18, which is optimistic: neighbouring frames are about
5 s apart, so test images have near-duplicates in training. The day-based split is the honest estimate.
Two ideas did not help: warping frames so the panel fills the image (test MAE 0.106) and the smaller
EfficientNet-B0 (same accuracy, slower on our GPU). See the report for details.

## Data (not included)

Download the DeepSolarEye dataset separately and point the code at the `PanelImages` folder:

```
# PowerShell
$env:SOLAR_DATA_ROOT = "C:\path\to\Solar_Panel_Soiling_Image_dataset\PanelImages"
```

`data.py` falls back to a default path on the author's PC if the variable is not set. Raw images are never
modified. The frozen splits are in `splits/` and must not be regenerated.

## Setup

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline (run from this folder)

| Step | Script | Output |
|---|---|---|
| 1 Dataset audit and manifest | `step1_manifest.py` | `step1_outputs/` |
| 2 Per-day summary, leakage-aware splits | `step2a_summary.py`, `step2_splits.py` | `splits/` |
| 3 Data pipeline, baseline | `data.py`, `baseline.py` | `baseline_results.csv`, `sanity_batch.png` |
| 4 ResNet-18 training | `train.py --strategy grouped` (also `random`, `--seed N`) | `outputs/` |
| 5 Evaluation, error analysis | `evaluate.py` | `outputs/eval/` |
| 6 Grad-CAM | `gradcam.py` | `outputs/gradcam_grouped/` |
| 7 Robustness and occlusion | `robustness.py` | `outputs/eval/robustness_*` |
| 8 Panel crop experiment | `panel_crop.py`, `train.py --crop`, `compare_crop.py` | `outputs/eval/crop_*` |
| 9 EfficientNet-B0 comparison | `train_arch.py --arch efficientnet_b0`, `compare_models.py`, `benchmark.py` | `outputs/eval/models_*`, `benchmark.csv` |

`train_original.py` and `data_original.py` are the versions from before the Step 8 changes, kept for reference.
`docs/` holds a short PDF write-up per step and the scripts that build them.

Model checkpoints (`*.pt`) are not in the repository; re-create them with the training commands above
(about 5 to 10 minutes per run on an RTX 5060 laptop GPU). Fixed seed 42 is the default.
