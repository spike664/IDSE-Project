"""Step 3: dataset, transforms and loaders (shared by everyone).

Each teammate only changes DATA_ROOT (or sets the SOLAR_DATA_ROOT env variable).
Target is L as a FRACTION (0-1). Convert to percent only when reporting.

Sanity check (from the solar-soiling folder):
    python data.py
"""
import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

DATA_ROOT = Path(os.environ.get(
    "SOLAR_DATA_ROOT",
    r"C:\Users\manie\Downloads\SolarPanelSoilingImageDataset"
    r"\Solar_Panel_Soiling_Image_dataset\PanelImages"))
SPLITS_CSV = Path("splits/splits.csv")
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet


def get_transform(train: bool):
    ops = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if train:   # light photometric jitter only: no flips, no heavy crops
        ops.append(T.ColorJitter(brightness=0.15, contrast=0.15))
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class SoilingDataset(Dataset):
    """strategy: 'random' or 'grouped'; subset: 'train' | 'val' | 'test'."""

    def __init__(self, strategy: str, subset: str, train_aug: bool = None,
                 splits_csv=SPLITS_CSV, data_root=DATA_ROOT, crop: bool = False):
        df = pd.read_csv(splits_csv)
        self.df = df[df[f"{strategy}_split"] == subset].reset_index(drop=True)
        self.root = Path(data_root)
        aug = (subset == "train") if train_aug is None else train_aug
        self.tf = get_transform(aug)
        # Step 8: optional homography warp so the panel fills the frame (panel_crop.py)
        self.cropper = None
        if crop:
            from panel_crop import PanelCropper
            self.cropper = PanelCropper()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["filename"]).convert("RGB")
        if self.cropper is not None:
            img = self.cropper(img, r["group_id"])
        return self.tf(img), torch.tensor(r["L"], dtype=torch.float32), r["filename"]


def make_loader(strategy, subset, batch_size=32, num_workers=4, shuffle=None):
    ds = SoilingDataset(strategy, subset)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(subset == "train") if shuffle is None else shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("DATA_ROOT:", DATA_ROOT, "| exists:", DATA_ROOT.exists())
    print("CUDA available:", torch.cuda.is_available())
    inv_mean = torch.tensor(MEAN).view(3, 1, 1)
    inv_std = torch.tensor(STD).view(3, 1, 1)

    for strategy in ("random", "grouped"):
        for subset in ("train", "val", "test"):
            ds = SoilingDataset(strategy, subset)
            print(f"{strategy:8s} {subset:5s}: {len(ds)} images")

    # 3.4 one batch: shape/target checks + visual check of train vs test transform
    loader = make_loader("grouped", "train", batch_size=16, num_workers=0)
    x, y, names = next(iter(loader))
    print("\nBatch tensor:", tuple(x.shape), "| target range:",
          float(y.min()), "to", float(y.max()), "| dtype:", y.dtype)
    assert x.shape[1:] == (3, IMG_SIZE, IMG_SIZE) and 0 <= y.min() and y.max() <= 1

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
    for j in range(8):
        img = (x[j] * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, j].imshow(img); axes[0, j].set_title(f"train L={y[j]:.3f}", fontsize=8)
        axes[0, j].axis("off")
    tds = SoilingDataset("grouped", "test")
    for j in range(8):
        xi, yi, _ = tds[j * (len(tds) // 8)]
        img = (xi * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1, j].imshow(img); axes[1, j].set_title(f"test L={yi:.3f}", fontsize=8)
        axes[1, j].axis("off")
    plt.tight_layout(); plt.savefig("sanity_batch.png", dpi=110)
    print("Saved sanity_batch.png")

    # 3.6 loading speed (2 workers, 320 images)
    for nw in (0, 4):
        ld = DataLoader(SoilingDataset("grouped", "train"), batch_size=32,
                        shuffle=True, num_workers=nw)
        t0 = time.time()
        for k, _ in enumerate(ld):
            if k == 10:
                break
        dt = time.time() - t0
        print(f"num_workers={nw}: 11 batches (352 images) in {dt:.1f}s "
              f"-> ~{352 / dt:.0f} img/s -> epoch of train ~{32158 / (352 / dt) / 60:.1f} min of loading")
