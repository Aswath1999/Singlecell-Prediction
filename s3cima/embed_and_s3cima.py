#!/usr/bin/env python3
# ============================================================
# Baseline + Marker-by-marker CNN embedding + MAIT SCIMa
# WITH logging, accuracy parsing, and global summary
# ============================================================

import os
import re
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda")

PATCH_SIZE = 128
HALF = PATCH_SIZE // 2
TILE_SIZE = 2048
GPU_BATCH_SIZE = 256
ARCSINH_COFACTOR = 5.0

NUM_MARKERS = 38
OUT_DIM = 512

ROOT_DATA = "/mnt/volumec/Aswath/processed_data/data"
CELLS_CSV = "/mnt/volumec/Aswath/selected_samples.csv"
NEEDED_IDS = "needed_cell_ids.csv"
CHANNELS_FILE = Path(ROOT_DATA) / "common_channels.txt"

EMB_OUTDIR = Path("Embeddingsands3cima")
SCIMA_OUTDIR = Path("s3cimamarkersmasked")
SUMMARY_CSV = Path("marker_ablation_summary.csv")

EMB_OUTDIR.mkdir(exist_ok=True)
SCIMA_OUTDIR.mkdir(exist_ok=True)

# ============================================================
# LOAD MARKER NAMES
# ============================================================
with open(CHANNELS_FILE) as f:
    MARKER_NAMES = [l.strip() for l in f if l.strip()]

assert len(MARKER_NAMES) == NUM_MARKERS, \
    f"Expected {NUM_MARKERS} markers, got {len(MARKER_NAMES)}"

# ============================================================
# MODEL
# ============================================================
from torchvision.models import resnet18, ResNet18_Weights

class MultiHeadResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.adapter = nn.Sequential(
            nn.Conv2d(38, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 1, bias=False),
        )

        for layer in [base.conv1, base.bn1, base.layer1]:
            for p in layer.parameters():
                p.requires_grad = False

        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.adapter(x)
        x = self.features(x)
        return torch.flatten(x, 1)

# ============================================================
# PATCH EXTRACTION
# ============================================================
@torch.no_grad()
def extract_patches(tile, centers):
    _, H, W = tile.shape

    centers = centers.to(dtype=tile.dtype)  # ensure same dtype as tile

    cx = centers[:, 0] / (W - 1) * 2 - 1
    cy = centers[:, 1] / (H - 1) * 2 - 1

    lin = torch.linspace(
        -HALF, HALF - 1, PATCH_SIZE,
        device=tile.device,
        dtype=tile.dtype
    )

    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    gx = gx / (W - 1) * 2
    gy = gy / (H - 1) * 2

    grid = torch.stack([gx, gy], dim=-1)[None]
    grid = grid + torch.stack([cx, cy], dim=1)[:, None, None, :]
    grid = grid.clamp(-1, 1)

    tile = tile.unsqueeze(0).expand(len(centers), -1, -1, -1)
    patches = F.grid_sample(tile, grid, align_corners=False)
    return torch.nan_to_num(patches)
# ============================================================
# ACCURACY PARSING
# ============================================================
def parse_accuracy(scima_dir: Path):
    log = scima_dir / "log.txt"
    if not log.exists():
        raise RuntimeError(f"No log.txt in {scima_dir}")

    txt = log.read_text()

    m = re.search(r"Accuracy score:\s*([0-9.]+)", txt)
    if m:
        return float(m.group(1))

    m = re.search(r"accuracy[:=]\s*([0-9.]+)", txt, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None
# ============================================================
# MAIN
# ============================================================
def main():

    # ---------- Load & filter cells ----------
    df = pd.read_csv(CELLS_CSV)
    df["cx"] = (df.XMin + df.XMax) / 2
    df["cy"] = (df.YMin + df.YMax) / 2
    df["global_cell_id"] = (
        df.Patient.astype(str) + "_" +
        df.cx.astype(int).astype(str) + "_" +
        df.cy.astype(int).astype(str)
    )

    needed = pd.read_csv(NEEDED_IDS)
    df = df.merge(needed, on="global_cell_id", how="inner")

    print(f"Using {len(df)} cells for all runs")

    # ---------- Model ----------
    model = MultiHeadResNet18().to(DEVICE).half().eval()

    # ---------- Runs: baseline + marker ablations ----------
    runs = [("baseline", -1)] + [
    (f"marker_{m:02d}_{MARKER_NAMES[m]}", m)
    for m in range(NUM_MARKERS)
]
    summary = []

    for tag, m in runs:
        print(f"\n🧪 RUN: {tag}")

        out_csv = EMB_OUTDIR / f"{tag}.csv"
        scima_dir = SCIMA_OUTDIR / tag
        scima_dir.mkdir(exist_ok=True, parents=True)

        # ====================================================
        # EMBEDDING
        # ====================================================
        if not out_csv.exists():
            write_header = True
            with open(out_csv, "w") as fout:

                for pid in tqdm(df.Patient.unique(), desc=f"Patients ({tag})", dynamic_ncols=True):
                    z = zarr.open(os.path.join(ROOT_DATA, str(pid), "data.zarr"), "r")
                    sdf = df[df.Patient == pid]

                    cell_pbar = tqdm(
                        total=len(sdf),
                        desc=f"Cells (P{pid})",
                        dynamic_ncols=True,
                        leave=False
                    )

                    for ty in range(0, z.shape[1], TILE_SIZE):
                        for tx in range(0, z.shape[2], TILE_SIZE):

                            ss = sdf[
                                (sdf.cx >= tx) & (sdf.cx < tx + TILE_SIZE) &
                                (sdf.cy >= ty) & (sdf.cy < ty + TILE_SIZE)
                            ]
                            if ss.empty:
                                continue

                            tile = torch.from_numpy(
                                z[:, ty:ty+TILE_SIZE, tx:tx+TILE_SIZE].astype(np.float32)
                            ).to(DEVICE).half()

                            centers = torch.stack([
                                torch.tensor(ss.cx.values - tx),
                                torch.tensor(ss.cy.values - ty)
                            ], dim=1).to(DEVICE)

                            for i in range(0, len(ss), GPU_BATCH_SIZE):
                                sub = centers[i:i+GPU_BATCH_SIZE]
                                idxs = ss.index[i:i+GPU_BATCH_SIZE]

                                patches = extract_patches(tile, sub)
                                patches = torch.arcsinh(patches / ARCSINH_COFACTOR)

                                if m != -1:
                                    patches[:, m] = 0.0

                                with torch.no_grad():
                                    emb = model(patches)

                                emb = emb.float().cpu().numpy()

                                for j, idx in enumerate(idxs):
                                    row = ss.loc[idx]
                                    meta = {
                                        "cell_index": idx,
                                        "Patient": row.Patient,
                                        "Tissue": row.Tissue,
                                        "Class0": row.Class0,
                                        "cx": row.cx,
                                        "cy": row.cy,
                                    }
                                    for d in range(OUT_DIM):
                                        meta[f"emb_{d}"] = emb[j, d]

                                    if write_header:
                                        fout.write(",".join(meta.keys()) + "\n")
                                        write_header = False
                                    fout.write(",".join(map(str, meta.values())) + "\n")

                                cell_pbar.update(len(idxs))
                                cell_pbar.set_postfix(
                                    speed=f"{cell_pbar.format_dict['rate']:.1f} cells/s"
                                )

                            del tile
                            torch.cuda.empty_cache()

                    cell_pbar.close()

        # ====================================================
        # SCIMa (LOGGED)
        # ====================================================
        log_file = scima_dir / "log.txt"
        if not log_file.exists():
            with open(log_file, "w") as lf:
                try:
                    subprocess.run(
                        [
                            "python", "s3_analysis.py",
                            "--data", str(out_csv),
                            "--out", str(scima_dir),
                            "--run_mait",
                            "--K", "50",
                            "--nrun", "10"
                        ],
                        stdout=lf,
                        stderr=lf,
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"❌ SCIMa failed for {tag}, skipping")
                    continue

        # ====================================================
        # PARSE ACCURACY
        # ====================================================
        acc = parse_accuracy(scima_dir)
        print("accuracy: ",acc)
     

        summary.append({
            "run": tag,
            "masked_marker_index": "none" if m == -1 else m,
            "masked_marker_name": "none" if m == -1 else MARKER_NAMES[m],
            "accuracy": acc
        })

        gc.collect()

    # ========================================================
    # SAVE SUMMARY
    # ========================================================
    pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)
    print(f"\n📊 Summary written to {SUMMARY_CSV}")

# ============================================================
if __name__ == "__main__":
    main()