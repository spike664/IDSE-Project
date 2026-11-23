"""Step 1: build dataset_manifest.csv and run sanity checks on DeepSolarEye.

Usage (Windows, from any folder):
    pip install pandas matplotlib pillow
    python step1_manifest.py

Edit IMG_DIR below if your path differs. The raw images are only read, never changed.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG_DIR = Path(r"C:\Users\manie\Downloads\SolarPanelSoilingImageDataset"
               r"\Solar_Panel_Soiling_Image_dataset\PanelImages")
OUT_DIR = Path("step1_outputs")
OUT_DIR.mkdir(exist_ok=True)

PAT = re.compile(
    r"^solar_(?P<dow>[A-Za-z]{3})_(?P<mon>[A-Za-z]{3})_(?P<day>\d+)_"
    r"(?P<h>\d+)__(?P<m>\d+)__(?P<s>\d+)_(?P<year>\d{4})_"
    r"L_(?P<L>[0-9eE+\-.]+?)_I_(?P<I>[0-9eE+\-.]+?)\.jpe?g$",
    re.IGNORECASE,
)

files = sorted(p for p in IMG_DIR.iterdir() if p.is_file())
print("Files in folder:", len(files))

rows, bad_names = [], []
for p in files:
    m = PAT.match(p.name)
    if not m:
        bad_names.append(p.name)
        continue
    d = m.groupdict()
    ts = pd.to_datetime(
        f"{d['year']} {d['mon']} {d['day']} {d['h']}:{d['m']}:{d['s']}",
        format="%Y %b %d %H:%M:%S", errors="coerce")
    rows.append({"filename": p.name, "path": str(p), "timestamp": ts,
                 "L": float(d["L"]), "I": float(d["I"])})

print("Parsed OK:", len(rows), "| filename parse failures:", len(bad_names))
if bad_names:
    print("First failures:", bad_names[:5])
if not rows:
    sys.exit("Nothing parsed - send me 5 raw filenames.")

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df["time_gap_sec"] = df["timestamp"].diff().dt.total_seconds()
print("Bad timestamps:", int(df["timestamp"].isna().sum()))

# Image readability + size (header read only for size, verify() for corruption)
corrupt, sizes = [], {}
for i, path in enumerate(df["path"]):
    try:
        with Image.open(path) as im:
            sizes[(im.size, im.mode)] = sizes.get((im.size, im.mode), 0) + 1
            im.verify()
    except Exception:
        corrupt.append(path)
    if (i + 1) % 10000 == 0:
        print("  checked", i + 1)
print("Corrupt/unreadable:", len(corrupt))
print("Image (size, mode) counts:", sizes)
df["corrupt"] = df["path"].isin(corrupt)

df.to_csv(OUT_DIR / "dataset_manifest.csv", index=False)

print("\n--- Label stats ---")
print(df[["L", "I"]].describe().round(4))
print("L median:", df["L"].median())

print("\n--- Time ---")
print("Date range:", df["timestamp"].min(), "->", df["timestamp"].max())
print("Distinct days:", df["timestamp"].dt.date.nunique())
print("Gap stats (sec):")
print(df["time_gap_sec"].describe().round(2))
for thr in (10, 60, 600, 3600):
    print(f"Gaps > {thr}s:", int((df["time_gap_sec"] > thr).sum()))

# Plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["L"], bins=60); ax[0].set_title("L (power loss field)")
ax[1].hist(df["I"], bins=60); ax[1].set_title("I (irradiance field)")
plt.tight_layout(); plt.savefig(OUT_DIR / "label_hist.png", dpi=120); plt.close()

# Sample grid: 12 images spread across the sorted L range
samp = df[~df["corrupt"]].sort_values("L").iloc[
    [int(i * (len(df) - 1) / 11) for i in range(12)]]
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
for a, (_, r) in zip(axes.ravel(), samp.iterrows()):
    a.imshow(Image.open(r["path"]).convert("RGB"))
    a.set_title(f"L={r['L']:.4f}  I={r['I']:.3f}", fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "sample_grid_by_L.png", dpi=110)
print("\nSaved to", OUT_DIR.resolve())
print("Sample filenames:", df["filename"].head(5).tolist())
