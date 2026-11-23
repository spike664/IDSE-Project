"""Step 8: panel standardisation by homography (future-work item from the survey).

Each frame is warped so that the soiled panel fills the image: the 4 panel corners
(top-left, top-right, bottom-right, bottom-left, in original 192x192 pixel
coordinates) are mapped to the corners of a 224x224 square. This removes the
neighbouring panel, the stand and the ground.

The camera framing shifts between sessions (some sessions are zoomed in or moved),
so corners are stored per session (group_id from splits/splits.csv). Sessions that
are not listed in crop_corners.csv use DEFAULT_CORNERS.

Workflow (from the solar-soiling folder, venv active):
    1. python panel_crop.py --preview
         -> outputs/crop/preview_*.png: one row per session (middle frame), original
            with the corner quad drawn + the warped result. Check every row.
    2. For any session whose quad does not fit the panel:
         python panel_crop.py --pick 7 12
         -> click the 4 panel corners (TL, TR, BR, BL) on that session's frame;
            the corners are saved to crop_corners.csv. Re-run --preview to confirm.
    3. Train with the crop:  python train.py --strategy grouped --crop
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CORNERS_CSV = Path("crop_corners.csv")
OUT_SIZE = 224
# TL, TR, BR, BL in original 192x192 pixels; estimated from the typical framing.
DEFAULT_CORNERS = [(78, 21), (191, 34), (159, 164), (17, 145)]
COLS = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]


def perspective_coeffs(src_corners, size=OUT_SIZE):
    """Coefficients for PIL's PERSPECTIVE transform. PIL maps each OUTPUT pixel
    (x, y) back to an INPUT pixel, so we solve the homography output -> input."""
    dst = [(0, 0), (size, 0), (size, size), (0, size)]
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src_corners):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()


def load_corners(path=CORNERS_CSV):
    """{group_id: [(x, y) x4]} for sessions with their own corners."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return {int(r.group_id): [(r[c], r[c.replace("_x", "_y")]) for c in COLS[::2]]
            for _, r in df.iterrows()}


class PanelCropper:
    def __init__(self, corners_csv=CORNERS_CSV, size=OUT_SIZE):
        self.size = size
        self.corners = load_corners(corners_csv)
        self.per_group = {g: perspective_coeffs(c, size) for g, c in self.corners.items()}
        self.default = perspective_coeffs(DEFAULT_CORNERS, size)

    def corners_for(self, group_id):
        return self.corners.get(int(group_id), DEFAULT_CORNERS)

    def __call__(self, img: Image.Image, group_id) -> Image.Image:
        coeffs = self.per_group.get(int(group_id), self.default)
        return img.transform((self.size, self.size), Image.PERSPECTIVE, coeffs,
                             Image.BICUBIC)


def session_frames(splits_csv="splits/splits.csv"):
    """Middle frame of every session, with its day, size and split membership."""
    df = pd.read_csv(splits_csv)
    rows = []
    for g, d in df.groupby("group_id"):
        d = d.sort_values("timestamp")
        r = d.iloc[len(d) // 2]
        rows.append({"group_id": int(g), "filename": r["filename"], "day": r["day"],
                     "n": len(d), "start": d["timestamp"].iloc[0],
                     "grouped_split": r["grouped_split"]})
    return pd.DataFrame(rows)


def preview(data_root, per_page=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crop = PanelCropper()
    sessions = session_frames()
    out = Path("outputs/crop"); out.mkdir(parents=True, exist_ok=True)
    custom = crop.corners
    for page, start in enumerate(range(0, len(sessions), per_page)):
        chunk = sessions.iloc[start:start + per_page]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6, 3 * len(chunk)))
        axes = np.atleast_2d(axes)
        for ax, (_, s) in zip(axes, chunk.iterrows()):
            img = Image.open(Path(data_root) / s["filename"]).convert("RGB")
            quad = img.copy()
            pts = crop.corners_for(s["group_id"])
            ImageDraw.Draw(quad).line(pts + [pts[0]], fill=(255, 0, 0), width=2)
            src = "custom" if s["group_id"] in custom else "default"
            ax[0].imshow(quad)
            ax[0].set_title(f"session {s['group_id']} | {s['day']} | n={s['n']} | "
                            f"{s['grouped_split']} | {src}", fontsize=7)
            ax[1].imshow(crop(img, s["group_id"])); ax[1].set_title("warped", fontsize=7)
            for a in ax:
                a.axis("off")
        plt.tight_layout()
        f = out / f"preview_{page + 1:02d}.png"
        plt.savefig(f, dpi=90); plt.close(fig)
        print("Saved", f)


def pick(data_root, group_ids):
    import matplotlib.pyplot as plt   # needs an interactive backend (default on Windows)

    sessions = session_frames().set_index("group_id")
    df = pd.read_csv(CORNERS_CSV) if CORNERS_CSV.exists() else pd.DataFrame(columns=["group_id"] + COLS)
    for g in group_ids:
        s = sessions.loc[g]
        img = Image.open(Path(data_root) / s["filename"]).convert("RGB")
        plt.figure(figsize=(7, 7)); plt.imshow(img)
        plt.title(f"session {g}: click TL, TR, BR, BL panel corners")
        pts = plt.ginput(4, timeout=0); plt.close()
        if len(pts) != 4:
            print(f"session {g}: skipped (need 4 clicks)"); continue
        row = {"group_id": g, **{c: round(v, 1) for c, v in zip(COLS, np.ravel(pts))}}
        df = pd.concat([df[df["group_id"] != g], pd.DataFrame([row])], ignore_index=True)
        print(f"session {g}: {pts}")
    df.sort_values("group_id").to_csv(CORNERS_CSV, index=False)
    print("Saved", CORNERS_CSV)


if __name__ == "__main__":
    from data import DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--pick", type=int, nargs="+", metavar="GROUP_ID")
    a = ap.parse_args()
    if a.pick:
        pick(DATA_ROOT, a.pick)
    if a.preview or not a.pick:
        preview(DATA_ROOT)
