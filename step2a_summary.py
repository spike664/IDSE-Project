"""Step 2.1 + 2.2: per-day label summary and session/group sizes.
Run from the solar-soiling folder:  python step2a_summary.py
Needs step1_outputs/dataset_manifest.csv (made by step1_manifest.py).
"""
import pandas as pd

d = pd.read_csv("step1_outputs/dataset_manifest.csv", parse_dates=["timestamp"])
d = d.sort_values("timestamp").reset_index(drop=True)
d["day"] = d["timestamp"].dt.date

print("=== 2.1 Per-day summary ===")
day = d.groupby("day").agg(
    n=("L", "size"),
    L_mean=("L", "mean"),
    L_median=("L", "median"),
    L_min=("L", "min"),
    L_max=("L", "max"),
    I_mean=("I", "mean"),
    first=("timestamp", "min"),
    last=("timestamp", "max"),
).round(3)
print(day.to_string())

print("\n=== 2.2 Groups (new group when gap > 600 s) ===")
d["group_id"] = (d["time_gap_sec"].fillna(0) > 600).cumsum()
g = d.groupby("group_id").agg(
    n=("L", "size"),
    L_mean=("L", "mean"),
    start=("timestamp", "min"),
    end=("timestamp", "max"),
)
print("Number of groups:", len(g))
print("Group size min / median / max:", g["n"].min(), g["n"].median(), g["n"].max())
print("\nLargest 8 groups:")
print(g.sort_values("n", ascending=False).head(8).round(3).to_string())
print("\nSmallest 5 groups:")
print(g.sort_values("n").head(5).round(3).to_string())
