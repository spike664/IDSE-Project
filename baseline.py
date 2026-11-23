"""Step 3.5: mean-predictor baseline. No torch needed.

Predicts the TRAIN mean of L for every val/test image. These are the numbers any CNN
must beat. Run from the solar-soiling folder:   python baseline.py
"""
import numpy as np
import pandas as pd

df = pd.read_csv("splits/splits.csv")
BINS = [(0.0, 0.15, "low  L<0.15"), (0.15, 0.5, "mid  0.15-0.5"), (0.5, 1.01, "high L>=0.5")]


def metrics(y, p):
    err = y - p
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    r2 = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return mae, rmse, r2


rows = []
for strategy in ("random", "grouped"):
    col = f"{strategy}_split"
    mean_L = df.loc[df[col] == "train", "L"].mean()
    print(f"\n=== {strategy} split | train mean L = {mean_L:.4f} ===")
    for subset in ("val", "test"):
        y = df.loc[df[col] == subset, "L"].to_numpy()
        mae, rmse, r2 = metrics(y, np.full_like(y, mean_L))
        rows.append((strategy, subset, len(y), mae, rmse, r2))
        print(f"{subset:4s} n={len(y):5d} | MAE {mae:.4f} ({mae*100:.2f} pts) | "
              f"RMSE {rmse:.4f} | R2 {r2:+.4f}")
        for lo, hi, name in BINS:
            m = (y >= lo) & (y < hi)
            if m.any():
                print(f"       {name}: n={m.sum():5d}  MAE {np.abs(y[m] - mean_L).mean():.4f}")

out = pd.DataFrame(rows, columns=["strategy", "set", "n", "MAE", "RMSE", "R2"]).round(4)
out.to_csv("baseline_results.csv", index=False)
print("\nSaved baseline_results.csv")
