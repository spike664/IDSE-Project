"""Step 8b: compare the original ResNet-18 with the panel-cropped one (same split, same settings).

Run after training both (from the solar-soiling folder, venv active):
    python compare_crop.py                 # grouped split (default)
    python compare_crop.py --strategy random

Reads outputs/preds/resnet18_<strategy>[_crop][_seedN]_{val,test}.csv. If extra seeds were
trained (python train.py --strategy grouped --crop --seed 1), every seed is listed and the
mean and spread per variant are reported, which shows whether a difference is bigger than
seed-to-seed noise.
Writes outputs/eval/crop_comparison.csv, crop_per_range.csv, crop_per_day.csv, crop_scatter.png.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [(0.0, 0.15, "low"), (0.15, 0.5, "mid"), (0.5, 1.01, "high")]
PREDS, OUT = Path("outputs/preds"), Path("outputs/eval")


def scores(y, p):
    e = p - y
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), "bias": e.mean()}


def find_runs(strategy, subset):
    """{(variant, seed): DataFrame} for the plain and cropped runs."""
    pat = re.compile(rf"resnet18_{strategy}(_crop)?(?:_seed(\d+))?_{subset}\.csv$")
    runs = {}
    for f in sorted(PREDS.glob(f"resnet18_{strategy}*_{subset}.csv")):
        m = pat.search(f.name)
        if m:   # skips debug runs such as _lim2000
            variant = "cropped" if m.group(1) else "original"
            runs[(variant, int(m.group(2) or 42))] = pd.read_csv(f)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="grouped", choices=["grouped", "random"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.read_csv("splits/splits.csv", usecols=["filename", "day"])

    rows, per_range, per_day = [], [], []
    for subset in ("val", "test"):
        runs = find_runs(a.strategy, subset)
        if not any(v == "cropped" for v, _ in runs):
            raise SystemExit(f"No cropped {subset} predictions found in {PREDS}; "
                             f"run: python train.py --strategy {a.strategy} --crop")
        for (variant, seed), df in runs.items():
            y, p = df["L_true"].values, df["L_pred"].values
            rows.append({"split": a.strategy, "set": subset, "variant": variant,
                         "seed": seed, "n": len(y), **scores(y, p)})
            if subset != "test":
                continue
            for lo, hi, name in BINS:
                m = (y >= lo) & (y < hi)
                sc = scores(y[m], p[m]); sc.pop("R2")   # R2 is meaningless inside a narrow range
                per_range.append({"variant": variant, "seed": seed, "range": name,
                                  "n": int(m.sum()), **sc})
            d = df.merge(days, on="filename")
            for day, g in d.groupby("day"):
                per_day.append({"variant": variant, "seed": seed, "day": day, "n": len(g),
                                **scores(g["L_true"].values, g["L_pred"].values)})

    res = pd.DataFrame(rows).sort_values(["set", "variant", "seed"])
    res.round(4).to_csv(OUT / "crop_comparison.csv", index=False)
    pd.DataFrame(per_range).round(4).to_csv(OUT / "crop_per_range.csv", index=False)
    pd.DataFrame(per_day).round(4).to_csv(OUT / "crop_per_day.csv", index=False)

    pd.set_option("display.width", 140)
    print(f"\n{a.strategy} split, each run (seed 42 = the original reported setting):")
    print(res.round(4).to_string(index=False))
    summ = res.groupby(["set", "variant"]).agg(
        runs=("seed", "count"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        R2=("R2", "mean"), R2_sd=("R2", "std"))
    print("\nMean over seeds (sd is blank with one seed):")
    print(summ.round(4).to_string())
    print("\nTest MAE by L range:")
    print(pd.DataFrame(per_range).round(4).to_string(index=False))
    print("\nTest MAE by day:")
    print(pd.DataFrame(per_day).round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = find_runs(a.strategy, "test")
    pair = [k for k in (("original", 42), ("cropped", 42)) if k in test]
    fig, axes = plt.subplots(1, len(pair), figsize=(5 * len(pair), 4.6), squeeze=False)
    for ax, k in zip(axes[0], pair):
        df = test[k]
        s = scores(df["L_true"].values, df["L_pred"].values)
        hb = ax.hexbin(df["L_true"], df["L_pred"], gridsize=40, bins="log",
                       extent=(0, 1, 0, 1), mincnt=1)
        ax.plot([0, 1], [0, 1], "r--", lw=1)
        ax.set(xlabel="true L", ylabel="predicted L", xlim=(0, 1), ylim=(0, 1),
               title=f"{k[0]} (seed {k[1]}): MAE {s['MAE']:.4f}, R2 {s['R2']:.3f}")
        fig.colorbar(hb, ax=ax, label="log10 count")
    plt.tight_layout(); plt.savefig(OUT / "crop_scatter.png", dpi=120)
    print("\nSaved", ", ".join(f"outputs/eval/{n}" for n in (
        "crop_comparison.csv", "crop_per_range.csv", "crop_per_day.csv", "crop_scatter.png")))


if __name__ == "__main__":
    main()
