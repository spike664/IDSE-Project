"""Step 9c: size and speed of the two models (parameters, checkpoint size, inference time).

    python benchmark.py                 # uses outputs/checkpoints/<arch>_grouped_best.pt if present
    python benchmark.py --repeats 30

Times are for one forward pass on a random 224x224 input (no data loading), after warm-up, median of
--repeats runs, reported per image. GPU: fp32 and bf16 (bf16 is what train/eval used); CPU: fp32.
Run it with nothing else heavy running, and quote the machine in the report. Weights are random if no
checkpoint exists (speed does not depend on the weights).
Writes outputs/eval/benchmark.csv.
"""
import argparse
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from train_arch import ARCHS, build_model


def timed(model, x, device, bf16, repeats, warmup=10):
    def fwd():
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            model(x)
    for _ in range(warmup):
        fwd()
    ts = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000 / x.shape[0]      # ms per image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--strategy", default="grouped")
    a = ap.parse_args()
    torch.manual_seed(42)
    print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    rows = []
    for arch in ARCHS:
        model = build_model(arch, pretrained=False)
        ck = Path(f"outputs/checkpoints/{arch}_{a.strategy}_best.pt")
        if ck.exists():
            model.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
        model.eval()
        row = {"model": arch, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
               "checkpoint_MB": round(ck.stat().st_size / 1e6, 1) if ck.exists() else None}
        settings = [("cpu", False)] + ([("cuda", False), ("cuda", True)] if torch.cuda.is_available() else [])
        for dev_name, bf16 in settings:
            dev = torch.device(dev_name)
            m = model.to(dev)
            for bs in (1, 64):
                if dev_name == "cpu" and bs == 64:
                    reps = max(5, a.repeats // 6)       # CPU batch 64 is slow
                else:
                    reps = a.repeats
                x = torch.randn(bs, 3, 224, 224, device=dev)
                key = f"{dev_name}{'_bf16' if bf16 else '_fp32'}_bs{bs}_ms_per_img"
                row[key] = round(timed(m, x, dev, bf16, reps), 3)
                print(arch, key, row[key])
        rows.append(row)
        model.to("cpu")
    out = pd.DataFrame(rows)
    Path("outputs/eval").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/eval/benchmark.csv", index=False)
    print("\n" + out.T.to_string())
    print("\nSaved outputs/eval/benchmark.csv")


if __name__ == "__main__":
    main()
