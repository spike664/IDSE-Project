"""Step 9b: compare two model families on the same split and input (e.g. ResNet-18 vs EfficientNet-B0).

    python compare_models.py                                  # resnet18 vs efficientnet_b0, grouped, original frames
    python compare_models.py --a resnet18 --b efficientnet_b0 --strategy grouped [--crop]

Reads outputs/preds/<arch>_<strategy>[_crop][_seedN][_lrX]_{val,test}.csv. Only runs with the same
input (--crop or not) are compared, so cropped and uncropped runs are never mixed. A run trained with
a changed learning rate (_lrX suffix) becomes its own family, e.g. efficientnet_b0_lr3e-05.
Writes outputs/eval/models_comparison.csv, models_per_range.csv, models_per_day.csv, models_scatter.png.
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


def find_runs(archs, strategy, subset, crop):
    """{(family, seed): DataFrame}; family = arch, plus _lrX if the lr was changed."""
    pat = re.compile(rf"^({'|'.join(archs)})_{strategy}(_crop)?(?:_seed(\d+))?(_lr[0-9.e+-]+)?_{subset}\.csv$")
    runs = {}
    for f in sorted(PREDS.glob(f"*_{strategy}*_{subset}.csv")):
        m = pat.match(f.name)
        if m and bool(m.group(2)) == crop:      # skips debug runs such as _lim2000
            runs[(m.group(1) + (m.group(4) or ""), int(m.group(3) or 42))] = pd.read_csv(f)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="resnet18")
    ap.add_argument("--b", default="efficientnet_b0")
    ap.add_argument("--strategy", default="grouped", choices=["grouped", "random"])
    ap.add_argument("--crop", action="store_true", help="compare the cropped-input runs")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.read_csv("splits/splits.csv", usecols=["filename", "day"])
    archs = [a.a, a.b]

    rows, per_range, per_day = [], [], []
    for subset in ("val", "test"):
        runs = find_runs(archs, a.strategy, subset, a.crop)
        fams = {f for f, _ in runs}
        if not any(f.startswith(a.b) for f in fams) or not any(f.startswith(a.a) for f in fams):
            raise SystemExit(f"Need {a.a} and {a.b} {subset} predictions in {PREDS} "
                             f"(found: {sorted(fams)}). Train with train_arch.py first.")
        for (fam, seed), df in runs.items():
            y, p = df["L_true"].values, df["L_pred"].values
            rows.append({"split": a.strategy, "set": subset, "model": fam, "seed": seed,
                         "n": len(y), **scores(y, p)})
            if subset != "test":
                continue
            for lo, hi, name in BINS:
                m = (y >= lo) & (y < hi)
                sc = scores(y[m], p[m]); sc.pop("R2")
                per_range.append({"model": fam, "seed": seed, "range": name, "n": int(m.sum()), **sc})
            d = df.merge(days, on="filename")
            for day, g in d.groupby("day"):
                per_day.append({"model": fam, "seed": seed, "day": day, "n": len(g),
                                **scores(g["L_true"].values, g["L_pred"].values)})

    res = pd.DataFrame(rows).sort_values(["set", "model", "seed"])
    res.round(4).to_csv(OUT / "models_comparison.csv", index=False)
    pr, pdy = pd.DataFrame(per_range), pd.DataFrame(per_day)
    pr.round(4).to_csv(OUT / "models_per_range.csv", index=False)
    pdy.round(4).to_csv(OUT / "models_per_day.csv", index=False)

    pd.set_option("display.width", 160)
    inp = "cropped" if a.crop else "original"
    print(f"\n{a.strategy} split, {inp} frames, each run:")
    print(res.round(4).to_string(index=False))
    summ = res.groupby(["set", "model"]).agg(runs=("seed", "count"), MAE=("MAE", "mean"),
                                             MAE_sd=("MAE", "std"), R2=("R2", "mean"), R2_sd=("R2", "std"))
    print("\nMean over seeds:")
    print(summ.round(4).to_string())
    t = summ.loc["test"]
    if a.a in t.index and a.b in t.index:
        gap = t.loc[a.b, "MAE"] - t.loc[a.a, "MAE"]
        sd = max(t.loc[a.a, "MAE_sd"], t.loc[a.b, "MAE_sd"])
        print(f"\nTest MAE gap ({a.b} minus {a.a}): {gap:+.4f}; larger seed sd: {sd:.4f}; "
              f"gap / sd = {gap / sd:+.1f}  (rule of thumb: only a clear difference if well above 1-2)")
    print("\nTest MAE by L range (mean over seeds):")
    print(pr.groupby(["model", "range"])[["MAE", "bias"]].mean().round(4).to_string())
    print("\nTest MAE by day (mean over seeds):")
    print(pdy.groupby(["model", "day"])[["MAE", "bias"]].mean().round(4).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = find_runs(archs, a.strategy, "test", a.crop)
    pair = [k for k in test if k[1] == 42]
    fig, axes = plt.subplots(1, max(len(pair), 1), figsize=(5 * max(len(pair), 1), 4.6), squeeze=False)
    for ax, k in zip(axes[0], pair):
        df = test[k]; s = scores(df["L_true"].values, df["L_pred"].values)
        hb = ax.hexbin(df["L_true"], df["L_pred"], gridsize=40, bins="log", extent=(0, 1, 0, 1), mincnt=1)
        ax.plot([0, 1], [0, 1], "r--", lw=1)
        ax.set(xlabel="true L", ylabel="predicted L", xlim=(0, 1), ylim=(0, 1),
               title=f"{k[0]} (seed 42): MAE {s['MAE']:.4f}, R2 {s['R2']:.3f}")
        fig.colorbar(hb, ax=ax, label="log10 count")
    plt.tight_layout(); plt.savefig(OUT / "models_scatter.png", dpi=120)
    print("\nSaved outputs/eval/models_comparison.csv, models_per_range.csv, models_per_day.csv, models_scatter.png")


if __name__ == "__main__":
    main()
