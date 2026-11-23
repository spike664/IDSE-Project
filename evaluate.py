"""Step 5: evaluation tables and plots from the saved predictions. No GPU/torch needed.

Run from the solar-soiling folder:   python evaluate.py
Reads  outputs/preds/resnet18_{random,grouped}_{val,test}.csv, splits/splits.csv
Writes outputs/eval/ : comparison_test.csv, per_day_error_*.csv, worst20_*.csv,
                       scatter_*.png, comparison_bar.png, error_by_range.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/eval")
OUT.mkdir(parents=True, exist_ok=True)
sp = pd.read_csv("splits/splits.csv")[["filename", "day", "I"]]
BINS = [(0.0, 0.15, "low"), (0.15, 0.5, "mid"), (0.5, 1.01, "high")]


def m(y, p):
    e = y - p
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}


rows, runs = [], {}
for strat in ("random", "grouped"):
    full = pd.read_csv("splits/splits.csv")
    col = f"{strat}_split"
    train_mean = full.loc[full[col] == "train", "L"].mean()
    for s in ("val", "test"):
        f = Path(f"outputs/preds/resnet18_{strat}_{s}.csv")
        if not f.exists():
            print("missing", f); continue
        d = pd.read_csv(f).merge(sp, on="filename")
        d["err"] = d["L_pred"] - d["L_true"]
        runs[(strat, s)] = d
        y, p = d["L_true"].to_numpy(), d["L_pred"].to_numpy()
        base = m(y, np.full_like(y, train_mean))
        cnn = m(y, p)
        rows.append({"split": strat, "set": s, "n": len(d),
                     "base_MAE": base["MAE"], "base_RMSE": base["RMSE"], "base_R2": base["R2"],
                     "CNN_MAE": cnn["MAE"], "CNN_RMSE": cnn["RMSE"], "CNN_R2": cnn["R2"],
                     "MAE_reduction_%": 100 * (1 - cnn["MAE"] / base["MAE"]),
                     "bias(pred-true)": d["err"].mean()})
tab = pd.DataFrame(rows).round(4)
tab.to_csv(OUT / "comparison_all.csv", index=False)
print("\n=== Baseline vs ResNet-18 (all sets) ===")
print(tab.to_string(index=False))
tab[tab["set"] == "test"].to_csv(OUT / "comparison_test.csv", index=False)

# per-range error and bias
print("\n=== Error by L range (test) ===")
rng_rows = []
for strat in ("random", "grouped"):
    d = runs.get((strat, "test"))
    if d is None:
        continue
    for lo, hi, name in BINS:
        mk = (d["L_true"] >= lo) & (d["L_true"] < hi)
        rng_rows.append({"split": strat, "range": name, "n": int(mk.sum()),
                         "MAE": d.loc[mk, "err"].abs().mean(), "bias": d.loc[mk, "err"].mean()})
rr = pd.DataFrame(rng_rows).round(4)
print(rr.to_string(index=False))
rr.to_csv(OUT / "error_by_range.csv", index=False)

# per-day error (grouped val + test days; random test days for contrast)
for (strat, s), d in runs.items():
    g = d.groupby("day").agg(n=("err", "size"), MAE=("err", lambda e: e.abs().mean()),
                             bias=("err", "mean"), L_true_mean=("L_true", "mean"),
                             L_pred_mean=("L_pred", "mean")).round(4)
    g.to_csv(OUT / f"per_day_error_{strat}_{s}.csv")
    if strat == "grouped":
        print(f"\n=== Per-day error: grouped {s} ===")
        print(g.to_string())

# worst predictions (for Grad-CAM failure cases) and best examples
for (strat, s), d in runs.items():
    if s != "test":
        continue
    d.assign(abs_err=d["err"].abs()).sort_values("abs_err", ascending=False).head(20)[
        ["filename", "day", "L_true", "L_pred", "err", "I"]].round(4).to_csv(
        OUT / f"worst20_{strat}_test.csv", index=False)

# scatter plots
for (strat, s), d in runs.items():
    if s != "test":
        continue
    plt.figure(figsize=(5, 5))
    plt.hexbin(d["L_true"], d["L_pred"], gridsize=50, bins="log", cmap="viridis", mincnt=1)
    plt.plot([0, 1], [0, 1], "r--", lw=1)
    plt.xlabel("true L"); plt.ylabel("predicted L")
    plt.title(f"{strat} split - test (n={len(d)})"); plt.colorbar(label="log10 count")
    plt.tight_layout(); plt.savefig(OUT / f"scatter_{strat}_test.png", dpi=120); plt.close()

# bar chart: baseline vs CNN, test MAE/RMSE
t = tab[tab["set"] == "test"]
if len(t):
    x = np.arange(len(t)); w = 0.2
    plt.figure(figsize=(7, 4))
    plt.bar(x - 1.5 * w, t["base_MAE"], w, label="baseline MAE")
    plt.bar(x - 0.5 * w, t["CNN_MAE"], w, label="ResNet-18 MAE")
    plt.bar(x + 0.5 * w, t["base_RMSE"], w, label="baseline RMSE")
    plt.bar(x + 1.5 * w, t["CNN_RMSE"], w, label="ResNet-18 RMSE")
    plt.xticks(x, t["split"] + " split"); plt.ylabel("error (fraction of L)")
    plt.title("Test error: mean baseline vs ResNet-18"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "comparison_bar.png", dpi=120); plt.close()

print("\nSaved to", OUT.resolve())
