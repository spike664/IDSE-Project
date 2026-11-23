"""Step 2.3-2.7: build random and grouped (by day) splits, verify them, save CSVs.

Run from the solar-soiling folder:   python step2_splits.py
Needs step1_outputs/dataset_manifest.csv. Raw images are never touched.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
GAP_SEC = 600            # new session when the gap between images exceeds this
BINS = np.linspace(0, 1.0001, 21)   # 20 bins over L for distribution matching
OUT = Path("splits")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("step1_outputs/dataset_manifest.csv", parse_dates=["timestamp"])
df = df[~df["corrupt"]].sort_values("timestamp").reset_index(drop=True)
df["day"] = df["timestamp"].dt.date.astype(str)
gap = df["timestamp"].diff().dt.total_seconds().fillna(0)
df["group_id"] = (gap > GAP_SEC).cumsum()          # session id, kept as metadata
N = len(df)
print(f"Images: {N} | days: {df['day'].nunique()} | sessions: {df['group_id'].nunique()}")

# ---------------------------------------------------------------- 2.3 random
rng = np.random.RandomState(SEED)
perm = rng.permutation(N)
n_tr, n_va = int(0.70 * N), int(0.15 * N)
df["random_split"] = "test"
df.loc[perm[:n_tr], "random_split"] = "train"
df.loc[perm[n_tr:n_tr + n_va], "random_split"] = "val"

# ------------------------------------------------- 2.4 grouped split (by day)
# Why by day: images from one day share the same dust state, and soiling does
# NOT drift monotonically across the 15 days (cleaning/rain resets it), so a
# purely chronological split would put only high-L days in test. We keep each
# whole day in one set and pick the day-to-set assignment whose L distribution
# is closest between sets, with ~70/15/15 image counts.
days = sorted(df["day"].unique())
cnt = np.array([(df["day"] == d).sum() for d in days], dtype=float)
hist = np.array([np.histogram(df.loc[df["day"] == d, "L"], bins=BINS)[0]
                 for d in days], dtype=float)
total_hist = hist.sum(0)
idx = range(len(days))

def tv(h_a, h_b):
    return 0.5 * np.abs(h_a / h_a.sum() - h_b / h_b.sum()).sum()

best, best_cost = None, 1e9
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
                cost = (tv(hist[list(te)].sum(0), h_tr)
                        + tv(hist[list(va)].sum(0), h_tr)
                        + 2 * (abs(f_te - 0.15) + abs(f_va - 0.15)))
                if cost < best_cost:
                    best, best_cost = (te, va), cost

te, va = best
day_split = {d: "train" for d in days}
for i in te:
    day_split[days[i]] = "test"
for i in va:
    day_split[days[i]] = "val"
df["grouped_split"] = df["day"].map(day_split)
print("\nGrouped split chosen (by whole days):")
for s in ("train", "val", "test"):
    print(f"  {s:5s}: {[d for d in days if day_split[d] == s]}")

# ------------------------------------------------------------- 2.5 verify
print("\n=== Verification ===")
rows = []
for col in ("random_split", "grouped_split"):
    for s in ("train", "val", "test"):
        sub = df[df[col] == s]
        rows.append({"strategy": col, "set": s, "n": len(sub),
                     "pct": round(100 * len(sub) / N, 1),
                     "L_mean": round(sub["L"].mean(), 3),
                     "L_median": round(sub["L"].median(), 3),
                     "L_std": round(sub["L"].std(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# no day appears in two sets for the grouped split; no file in two sets at all
g_days = df.groupby("day")["grouped_split"].nunique()
assert (g_days == 1).all(), "a day appears in more than one grouped set"
assert df["filename"].is_unique
print("\nCheck OK: every day belongs to exactly one grouped set; filenames unique.")

# ------------------------------------------------ 2.6 leakage evidence
ts = (df["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds()  # unit-safe seconds
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

print("\n=== Time (seconds) from each val/test image to its nearest TRAIN image ===")
for col in ("random_split", "grouped_split"):
    g = nearest_train_gap(col)
    for s in ("val", "test"):
        a = g[s]
        print(f"{col:14s} {s:4s}: median {np.median(a):9.0f}s | "
              f"90th pct {np.percentile(a, 90):9.0f}s | max {a.max():9.0f}s | "
              f"share within 10s: {(a <= 10).mean() * 100:5.1f}%")

# ------------------------------------------------------------ plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ("random_split", "grouped_split")):
    for s in ("train", "val", "test"):
        ax.hist(df.loc[df[col] == s, "L"], bins=40, density=True,
                alpha=0.45, label=s)
    ax.set_title(col + " - L distribution"); ax.set_xlabel("L"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "split_L_hist.png", dpi=120); plt.close()

# ------------------------------------------------------------ 2.7 save
# No absolute paths: each teammate keeps the dataset in a different folder.
cols = ["filename", "timestamp", "L", "I", "day", "group_id"]
df[cols + ["random_split", "grouped_split"]].to_csv(OUT / "splits.csv", index=False)
for col, name in (("random_split", "random"), ("grouped_split", "grouped")):
    for s in ("train", "val", "test"):
        df.loc[df[col] == s, cols].to_csv(OUT / f"{name}_{s}.csv", index=False)
print("\nSaved to", OUT.resolve(), ":", sorted(p.name for p in OUT.iterdir()))
