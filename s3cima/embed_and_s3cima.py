#!/usr/bin/env python3
# ============================================================
# Baseline-first + Marker-by-marker CNN embedding + S3-CIMA
# Strict execution order:
#   PHASE 1: Baseline embedding → baseline S3-CIMA
#   PHASE 2: Marker embeddings → marker S3-CIMA (parallel)
# ============================================================

import os
import re
import gc
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

# ============================================================
# CONFIG
# ============================================================
PATCH_SIZE = 128
HALF = PATCH_SIZE // 2
TILE_SIZE = 2048
GPU_BATCH_SIZE = 256
ARCSINH_COFACTOR = 5.0

NUM_MARKERS = 38
OUT_DIM = 512

ROOT_DATA = "/mnt/volumec/Aswath/processed_data/data"
CELLS_CSV = "/mnt/volumec/Aswath/selected_samples.csv"
# NEEDED_IDS = "needed_cell_ids.csv"
CHANNELS_FILE = Path(ROOT_DATA) / "common_channels.txt"

EMB_OUTDIR = Path("Embeddingsands3cimafull")
SCIMA_OUTDIR = Path("s3cimamarkersmaskedfull")
SUMMARY_CSV = Path("marker_ablation_summary_full.csv")

EMB_OUTDIR.mkdir(exist_ok=True, parents=True)
SCIMA_OUTDIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# MODEL
# ============================================================
from torchvision.models import resnet18, ResNet18_Weights


class MultiHeadResNet18(nn.Module):
    def __init__(self, num_markers: int):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.adapter = nn.Sequential(
            nn.Conv2d(num_markers, 32, 1, bias=False),
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
    centers = centers.to(dtype=tile.dtype)

    cx = centers[:, 0] / (W - 1) * 2 - 1
    cy = centers[:, 1] / (H - 1) * 2 - 1

    lin = torch.linspace(
        -HALF, HALF - 1, PATCH_SIZE,
        device=tile.device, dtype=tile.dtype
    )
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    gx = gx / (W - 1) * 2
    gy = gy / (H - 1) * 2

    grid = torch.stack([gx, gy], dim=-1)[None]
    grid = grid + torch.stack([cx, cy], dim=1)[:, None, None, :]
    grid = grid.clamp(-1, 1)

    tileN = tile.unsqueeze(0).expand(len(centers), -1, -1, -1)
    patches = F.grid_sample(tileN, grid, align_corners=False)
    return torch.nan_to_num(patches)


# ============================================================
# HELPERS
# ============================================================
def safe_tag(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name.strip())
    return re.sub(r"_+", "_", name)[:120]


def load_cells(needed_ids_path: str | None = None):
    df = pd.read_csv(CELLS_CSV)

    # centers
    df["cx"] = (df.XMin + df.XMax) / 2
    df["cy"] = (df.YMin + df.YMax) / 2

    # global id (must match how you created needed_cell_ids.csv)
    df["global_cell_id"] = (
        df.Patient.astype(str) + "_" +
        df["cx"].astype(int).astype(str) + "_" +
        df["cy"].astype(int).astype(str)
    )

    # If you want FULL CSV, do NOT filter
    if needed_ids_path is None:
        print(f"✅ Using FULL CSV (no needed_cell_ids filter): {len(df)} cells")
        return df

    # Otherwise filter using needed_cell_ids.csv
    needed = pd.read_csv(needed_ids_path)

    # robust: accept different column names
    if "global_cell_id" in needed.columns:
        key = "global_cell_id"
    elif "cell_id" in needed.columns:
        key = "cell_id"
    else:
        key = needed.columns[0]
        print(f"⚠️ Using '{key}' as ID column in {needed_ids_path}")

    needed = needed.rename(columns={key: "global_cell_id"})[["global_cell_id"]].drop_duplicates()

    df_f = df.merge(needed, on="global_cell_id", how="inner")
    print(f"✅ Using FILTERED needed_cell_ids: {len(df_f)} / {len(df)} cells")
    return df_f


def parse_accuracy(scima_dir: Path):
    log = scima_dir / "log.txt"
    if not log.exists():
        return None
    txt = log.read_text(errors="ignore")
    m = re.search(r"Accuracy score:\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


# ============================================================
# EMBEDDING
# ============================================================
def write_embeddings(df, tag, marker_idx, device, model):
    out_csv = EMB_OUTDIR / f"{tag}.csv"
    if out_csv.exists():
        print(f"⏭️  Skipping existing embeddings: {tag}")
        return

    print(f"\n🧬 Embedding: {tag}")

    tmp = out_csv.with_suffix(".tmp.csv")
    write_header = True

    total_cells = len(df)
    cell_pbar = tqdm(
        total=total_cells,
        desc=f"Cells ({tag})",
        dynamic_ncols=True
    )

    with open(tmp, "w") as f:
        for pid in df.Patient.unique():
            z = zarr.open(os.path.join(ROOT_DATA, str(pid), "data.zarr"), "r")
            sdf = df[df.Patient == pid]

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
                    ).to(device).half()

                    centers = torch.stack([
                        torch.tensor(ss.cx.values - tx),
                        torch.tensor(ss.cy.values - ty)
                    ], dim=1).to(device)

                    for i in range(0, len(ss), GPU_BATCH_SIZE):
                        sub = centers[i:i+GPU_BATCH_SIZE]
                        idxs = ss.index[i:i+GPU_BATCH_SIZE]

                        patches = extract_patches(tile, sub)
                        patches = torch.arcsinh(patches / ARCSINH_COFACTOR)

                        # 🔴 marker masking
                        if marker_idx != -1:
                            patches[:, marker_idx] = 0.0

                        # 🔒 NO GRADIENTS
                        with torch.no_grad():
                            emb = model(patches)

                        emb = emb.detach().float().cpu().numpy()

                        for j, idx in enumerate(idxs):
                            row = ss.loc[idx]
                            meta = {
                                "cell_index": int(idx),
                                "Patient": row.Patient,
                                "Tissue": row.Tissue,
                                "Class0": row.Class0,
                                "cx": float(row.cx),
                                "cy": float(row.cy),
                            }
                            for d in range(OUT_DIM):
                                meta[f"emb_{d}"] = emb[j, d]

                            if write_header:
                                f.write(",".join(meta.keys()) + "\n")
                                write_header = False
                            f.write(",".join(map(str, meta.values())) + "\n")

                        # ✅ update PER CELL
                        cell_pbar.update(len(idxs))

                    del tile
                    torch.cuda.empty_cache()

    cell_pbar.close()
    tmp.replace(out_csv)
    print(f"✅ Wrote {out_csv}")
# ============================================================
# S3-CIMA
# ============================================================
def run_scima(task):
    out_csv, scima_dir, K, nrun = task
    scima_dir.mkdir(parents=True, exist_ok=True)
    log = scima_dir / "log.txt"
    if log.exists():
        return scima_dir.name, parse_accuracy(scima_dir)

    cmd = [
        "python", "s3_analysis.py",
        "--data", str(out_csv),
        "--out", str(scima_dir),
        "--K", str(K),
        "--nrun", str(nrun),
        "--run_mait",
    ]

    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=f, check=True)

    return scima_dir.name, parse_accuracy(scima_dir)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--nrun", type=int, default=10)
    parser.add_argument("--scima_workers", type=int, default=6)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    marker_names = [l.strip() for l in open(CHANNELS_FILE)]
    assert len(marker_names) == NUM_MARKERS

    df = load_cells(needed_ids_path=None)
    model = MultiHeadResNet18(NUM_MARKERS).to(device).half().eval()

    # --------------------------------------------------------
    # DEFINE RUNS
    # --------------------------------------------------------
    baseline_run = [("baseline", -1)]
    marker_runs = [
        (f"marker_{i:02d}_{safe_tag(marker_names[i])}", i)
        for i in range(NUM_MARKERS)
    ]

    scima_root = SCIMA_OUTDIR / f"K{args.K}_nrun{args.nrun}"
    scima_root.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # PHASE 1 — BASELINE
    # ========================================================
    print("\n================ PHASE 1: BASELINE ================")
    tag, m = baseline_run[0]
    write_embeddings(df, tag, m, device, model)

    _, baseline_acc = run_scima(
        (EMB_OUTDIR / f"{tag}.csv", scima_root / tag, args.K, args.nrun)
    )

    print(f"\n✅ BASELINE ACCURACY = {baseline_acc:.4f}")
    print("🚦 Baseline finished successfully. Proceeding to marker ablations.")

    # ========================================================
    # PHASE 2 — MARKER EMBEDDINGS
    # ========================================================
    print("\n================ PHASE 2: MARKER EMBEDDINGS ================")
    for tag, m in marker_runs:
        write_embeddings(df, tag, m, device, model)
        gc.collect()

    # ========================================================
    # PHASE 2 — MARKER S3-CIMA (PARALLEL)
    # ========================================================
    print("\n================ PHASE 2: MARKER S3-CIMA ================")

    tasks = [
        (EMB_OUTDIR / f"{tag}.csv", scima_root / tag, args.K, args.nrun)
        for tag, _ in marker_runs
    ]

    with Pool(processes=args.scima_workers) as pool:
        results = list(tqdm(pool.imap_unordered(run_scima, tasks), total=len(tasks)))

    # ========================================================
    # SUMMARY
    # ========================================================
    summary = [{
        "run": "baseline",
        "masked_marker_index": "none",
        "masked_marker_name": "none",
        "K": args.K,
        "nrun": args.nrun,
        "accuracy": baseline_acc
    }]

    tag_to_idx = dict(marker_runs)
    for tag, acc in results:
        m = tag_to_idx[tag]
        summary.append({
            "run": tag,
            "masked_marker_index": m,
            "masked_marker_name": marker_names[m],
            "K": args.K,
            "nrun": args.nrun,
            "accuracy": acc,
        })

    out = SUMMARY_CSV.with_name(
        f"{SUMMARY_CSV.stem}_K{args.K}_nrun{args.nrun}.csv"
    )
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\n📊 FINAL SUMMARY written to {out}")


if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# # ============================================================
# # Baseline + Marker-by-marker CNN embedding + S3-CIMA
# # Robust version:
# #  - Computes missing embeddings automatically
# #  - Reuses existing CSVs
# #  - Runs S3-CIMA in parallel on CPU
# #  - Writes per-(K,nrun) summaries
# # ============================================================

# import os
# import re
# import gc
# import argparse
# import subprocess
# from pathlib import Path
# from multiprocessing import Pool

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import zarr

# # ============================================================
# # CONFIG
# # ============================================================
# PATCH_SIZE = 128
# HALF = PATCH_SIZE // 2
# TILE_SIZE = 2048
# GPU_BATCH_SIZE = 256
# ARCSINH_COFACTOR = 5.0

# NUM_MARKERS = 38
# OUT_DIM = 512

# ROOT_DATA = "/mnt/volumec/Aswath/processed_data/data"
# CELLS_CSV = "/mnt/volumec/Aswath/selected_samples.csv"
# NEEDED_IDS = "needed_cell_ids.csv"
# CHANNELS_FILE = Path(ROOT_DATA) / "common_channels.txt"

# EMB_OUTDIR = Path("Embeddingsands3cima")
# SCIMA_OUTDIR = Path("s3cimamarkersmasked")
# SUMMARY_CSV = Path("marker_ablation_summary.csv")

# EMB_OUTDIR.mkdir(exist_ok=True, parents=True)
# SCIMA_OUTDIR.mkdir(exist_ok=True, parents=True)

# # ============================================================
# # MODEL
# # ============================================================
# from torchvision.models import resnet18, ResNet18_Weights


# class MultiHeadResNet18(nn.Module):
#     def __init__(self, num_markers: int):
#         super().__init__()
#         base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

#         self.adapter = nn.Sequential(
#             nn.Conv2d(num_markers, 32, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.GELU(),
#             nn.Conv2d(32, 3, 1, bias=False),
#         )

#         for layer in [base.conv1, base.bn1, base.layer1]:
#             for p in layer.parameters():
#                 p.requires_grad = False

#         self.features = nn.Sequential(*list(base.children())[:-1])

#     def forward(self, x):
#         x = self.adapter(x)
#         x = self.features(x)
#         return torch.flatten(x, 1)


# # ============================================================
# # PATCH EXTRACTION
# # ============================================================
# @torch.no_grad()
# def extract_patches(tile, centers):
#     _, H, W = tile.shape
#     centers = centers.to(dtype=tile.dtype)

#     cx = centers[:, 0] / (W - 1) * 2 - 1
#     cy = centers[:, 1] / (H - 1) * 2 - 1

#     lin = torch.linspace(-HALF, HALF - 1, PATCH_SIZE,
#                          device=tile.device, dtype=tile.dtype)
#     gx, gy = torch.meshgrid(lin, lin, indexing="ij")
#     gx = gx / (W - 1) * 2
#     gy = gy / (H - 1) * 2

#     grid = torch.stack([gx, gy], dim=-1)[None]
#     grid = grid + torch.stack([cx, cy], dim=1)[:, None, None, :]
#     grid = grid.clamp(-1, 1)

#     tileN = tile.unsqueeze(0).expand(len(centers), -1, -1, -1)
#     patches = F.grid_sample(tileN, grid, align_corners=False)
#     return torch.nan_to_num(patches)


# # ============================================================
# # HELPERS
# # ============================================================
# def safe_tag_name(name: str) -> str:
#     name = re.sub(r"[^\w.\-]+", "_", name.strip())
#     return re.sub(r"_+", "_", name)[:120]


# def build_runs(marker_names):
#     runs = [("baseline", -1)]
#     for m, name in enumerate(marker_names):
#         runs.append((f"marker_{m:02d}_{safe_tag_name(name)}", m))
#     return runs


# def load_cells():
#     df = pd.read_csv(CELLS_CSV)
#     df["cx"] = (df.XMin + df.XMax) / 2
#     df["cy"] = (df.YMin + df.YMax) / 2
#     df["global_cell_id"] = (
#         df.Patient.astype(str) + "_" +
#         df.cx.astype(int).astype(str) + "_" +
#         df.cy.astype(int).astype(str)
#     )
#     needed = pd.read_csv(NEEDED_IDS)
#     return df.merge(needed, on="global_cell_id", how="inner")


# # ============================================================
# # EMBEDDING
# # ============================================================
# def write_embeddings(df, tag, marker_idx, marker_names, device, model):
#     out_csv = EMB_OUTDIR / f"{tag}.csv"
#     if out_csv.exists():
#         return

#     print(f"🧬 Computing embeddings: {tag}")

#     tmp = out_csv.with_suffix(".tmp.csv")
#     write_header = True

#     # 🔥 NEW: global per-cell progress bar (one per marker)
#     total_cells = len(df)
#     cell_pbar = tqdm(
#         total=total_cells,
#         desc=f"Cells ({tag})",
#         dynamic_ncols=True
#     )

#     with open(tmp, "w") as f:
#         for pid in tqdm(df.Patient.unique(), desc=f"Patients ({tag})"):
#             z = zarr.open(os.path.join(ROOT_DATA, str(pid), "data.zarr"), "r")
#             sdf = df[df.Patient == pid]

#             for ty in range(0, z.shape[1], TILE_SIZE):
#                 for tx in range(0, z.shape[2], TILE_SIZE):
#                     ss = sdf[
#                         (sdf.cx >= tx) & (sdf.cx < tx + TILE_SIZE) &
#                         (sdf.cy >= ty) & (sdf.cy < ty + TILE_SIZE)
#                     ]
#                     if ss.empty:
#                         continue

#                     tile = torch.from_numpy(
#                         z[:, ty:ty + TILE_SIZE, tx:tx + TILE_SIZE].astype(np.float32)
#                     ).to(device).half()

#                     centers = torch.stack([
#                         torch.tensor(ss.cx.values - tx),
#                         torch.tensor(ss.cy.values - ty)
#                     ], dim=1).to(device)

#                     for i in range(0, len(ss), GPU_BATCH_SIZE):
#                         sub = centers[i:i + GPU_BATCH_SIZE]
#                         idxs = ss.index[i:i + GPU_BATCH_SIZE]

#                         patches = extract_patches(tile, sub)
#                         patches = torch.arcsinh(patches / ARCSINH_COFACTOR)

#                         if marker_idx != -1:
#                             patches[:, marker_idx] = 0.0

#                         with torch.no_grad():
#                             emb = model(patches)

#                         emb = emb.float().cpu().numpy()

#                         for j, idx in enumerate(idxs):
#                             row = ss.loc[idx]
#                             meta = {
#                                 "cell_index": int(idx),
#                                 "Patient": row.Patient,
#                                 "Tissue": row.Tissue,
#                                 "Class0": row.Class0,
#                                 "cx": float(row.cx),
#                                 "cy": float(row.cy),
#                             }
#                             for d in range(OUT_DIM):
#                                 meta[f"emb_{d}"] = emb[j, d]

#                             if write_header:
#                                 f.write(",".join(meta.keys()) + "\n")
#                                 write_header = False
#                             f.write(",".join(map(str, meta.values())) + "\n")

#                         # 🔥 NEW: update cell tqdm ONCE per batch
#                         cell_pbar.update(len(idxs))

#                     del tile
#                     torch.cuda.empty_cache()

#     cell_pbar.close()
#     tmp.replace(out_csv)

#     print(f"✅ Wrote embeddings: {out_csv}")
    
    
# # ============================================================
# # S3-CIMA
# # ============================================================
# def parse_accuracy(scima_dir):
#     log = scima_dir / "log.txt"
#     if not log.exists():
#         return None
#     txt = log.read_text(errors="ignore")
#     m = re.search(r"Accuracy score:\s*([0-9.]+)", txt)
#     return float(m.group(1)) if m else None


# def run_scima(args):
#     out_csv, scima_dir, K, nrun = args
#     scima_dir.mkdir(parents=True, exist_ok=True)
#     log = scima_dir / "log.txt"
#     if log.exists():
#         return scima_dir.name, parse_accuracy(scima_dir)

#     cmd = [
#         "python", "s3_analysis.py",
#         "--data", str(out_csv),
#         "--out", str(scima_dir),
#         "--K", str(K),
#         "--nrun", str(nrun),
#         "--run_mait",
#     ]
#     with open(log, "w") as f:
#         subprocess.run(cmd, stdout=f, stderr=f, check=True)

#     return scima_dir.name, parse_accuracy(scima_dir)


# # ============================================================
# # MAIN
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--embed_only", action="store_true")
#     parser.add_argument("--scima_only", action="store_true")
#     parser.add_argument("--full", action="store_true")
#     parser.add_argument("--gpu", default="0")
#     parser.add_argument("--K", type=int, default=50)
#     parser.add_argument("--nrun", type=int, default=10)
#     parser.add_argument("--scima_workers", type=int, default=6)
#     args = parser.parse_args()

#     if not (args.embed_only or args.scima_only or args.full):
#         args.full = True

#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     device = torch.device("cuda:0")

#     marker_names = [l.strip() for l in open(CHANNELS_FILE)]
#     runs = build_runs(marker_names)
#     df = load_cells()

#     model = MultiHeadResNet18(NUM_MARKERS).to(device).half().eval()

#     # ========================================================
#     # EMBEDDINGS (only missing)
#     # ========================================================
#     if args.embed_only or args.full or args.scima_only:
#         missing = [(t, m) for t, m in runs if not (EMB_OUTDIR / f"{t}.csv").exists()]
#         if missing:
#             print(f"\n🧬 Computing {len(missing)} missing embeddings")
#             for tag, m in missing:
#                 write_embeddings(df, tag, m, marker_names, device, model)
#                 gc.collect()
#         else:
#             print("\n✅ All embeddings present")

#     # ========================================================
#     # S3-CIMA
#     # ========================================================
#     if args.scima_only or args.full:
#         scima_root = SCIMA_OUTDIR / f"K{args.K}_nrun{args.nrun}"
#         scima_root.mkdir(parents=True, exist_ok=True)

#         tasks = []
#         for tag, _ in runs:
#             out_csv = EMB_OUTDIR / f"{tag}.csv"
#             if out_csv.exists():
#                 tasks.append((out_csv, scima_root / tag, args.K, args.nrun))

#         print(f"\n🧠 Running S3-CIMA on {len(tasks)} runs with {args.scima_workers} workers")

#         with Pool(processes=args.scima_workers) as pool:
#             results = list(tqdm(pool.imap_unordered(run_scima, tasks), total=len(tasks)))

#         summary = []
#         tag_to_m = dict(runs)
#         for tag, acc in results:
#             m = tag_to_m[tag]
#             summary.append({
#                 "run": tag,
#                 "masked_marker_index": "none" if m == -1 else m,
#                 "masked_marker_name": "none" if m == -1 else marker_names[m],
#                 "K": args.K,
#                 "nrun": args.nrun,
#                 "accuracy": acc,
#             })

#         out = SUMMARY_CSV.with_name(f"{SUMMARY_CSV.stem}_K{args.K}_nrun{args.nrun}.csv")
#         pd.DataFrame(summary).to_csv(out, index=False)
#         print(f"\n📊 Summary written to {out}")


# if __name__ == "__main__":
#     main()



# # #!/usr/bin/env python3
# # # ============================================================
# # # Baseline + Marker-by-marker CNN embedding + MAIT SCIMa
# # # WITH logging, accuracy parsing, and global summary
# # # ============================================================

# # import os
# # import re
# # import subprocess
# # import numpy as np
# # import pandas as pd
# # from pathlib import Path
# # from tqdm import tqdm
# # import gc

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import zarr

# # # ============================================================
# # # CONFIG
# # # ============================================================
# # DEVICE = torch.device("cuda")

# # PATCH_SIZE = 128
# # HALF = PATCH_SIZE // 2
# # TILE_SIZE = 2048
# # GPU_BATCH_SIZE = 256
# # ARCSINH_COFACTOR = 5.0

# # NUM_MARKERS = 38
# # OUT_DIM = 512

# # ROOT_DATA = "/mnt/volumec/Aswath/processed_data/data"
# # CELLS_CSV = "/mnt/volumec/Aswath/selected_samples.csv"
# # NEEDED_IDS = "needed_cell_ids.csv"
# # CHANNELS_FILE = Path(ROOT_DATA) / "common_channels.txt"

# # EMB_OUTDIR = Path("Embeddingsands3cima")
# # SCIMA_OUTDIR = Path("s3cimamarkersmasked")
# # SUMMARY_CSV = Path("marker_ablation_summary.csv")

# # EMB_OUTDIR.mkdir(exist_ok=True)
# # SCIMA_OUTDIR.mkdir(exist_ok=True)

# # # ============================================================
# # # LOAD MARKER NAMES
# # # ============================================================
# # with open(CHANNELS_FILE) as f:
# #     MARKER_NAMES = [l.strip() for l in f if l.strip()]

# # assert len(MARKER_NAMES) == NUM_MARKERS, \
# #     f"Expected {NUM_MARKERS} markers, got {len(MARKER_NAMES)}"

# # # ============================================================
# # # MODEL
# # # ============================================================
# # from torchvision.models import resnet18, ResNet18_Weights

# # class MultiHeadResNet18(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# #         self.adapter = nn.Sequential(
# #             nn.Conv2d(38, 32, 1, bias=False),
# #             nn.BatchNorm2d(32),
# #             nn.GELU(),
# #             nn.Conv2d(32, 3, 1, bias=False),
# #         )

# #         for layer in [base.conv1, base.bn1, base.layer1]:
# #             for p in layer.parameters():
# #                 p.requires_grad = False

# #         self.features = nn.Sequential(*list(base.children())[:-1])

# #     def forward(self, x):
# #         x = self.adapter(x)
# #         x = self.features(x)
# #         return torch.flatten(x, 1)

# # # ============================================================
# # # PATCH EXTRACTION
# # # ============================================================
# # @torch.no_grad()
# # def extract_patches(tile, centers):
# #     _, H, W = tile.shape

# #     centers = centers.to(dtype=tile.dtype)  # ensure same dtype as tile

# #     cx = centers[:, 0] / (W - 1) * 2 - 1
# #     cy = centers[:, 1] / (H - 1) * 2 - 1

# #     lin = torch.linspace(
# #         -HALF, HALF - 1, PATCH_SIZE,
# #         device=tile.device,
# #         dtype=tile.dtype
# #     )

# #     gx, gy = torch.meshgrid(lin, lin, indexing="ij")
# #     gx = gx / (W - 1) * 2
# #     gy = gy / (H - 1) * 2

# #     grid = torch.stack([gx, gy], dim=-1)[None]
# #     grid = grid + torch.stack([cx, cy], dim=1)[:, None, None, :]
# #     grid = grid.clamp(-1, 1)

# #     tile = tile.unsqueeze(0).expand(len(centers), -1, -1, -1)
# #     patches = F.grid_sample(tile, grid, align_corners=False)
# #     return torch.nan_to_num(patches)
# # # ============================================================
# # # ACCURACY PARSING
# # # ============================================================
# # def parse_accuracy(scima_dir: Path):
# #     log = scima_dir / "log.txt"
# #     if not log.exists():
# #         raise RuntimeError(f"No log.txt in {scima_dir}")

# #     txt = log.read_text()

# #     m = re.search(r"Accuracy score:\s*([0-9.]+)", txt)
# #     if m:
# #         return float(m.group(1))

# #     m = re.search(r"accuracy[:=]\s*([0-9.]+)", txt, re.IGNORECASE)
# #     if m:
# #         return float(m.group(1))

# #     return None
# # # ============================================================
# # # MAIN
# # # ============================================================
# # def main():

# #     # ---------- Load & filter cells ----------
# #     df = pd.read_csv(CELLS_CSV)
# #     df["cx"] = (df.XMin + df.XMax) / 2
# #     df["cy"] = (df.YMin + df.YMax) / 2
# #     df["global_cell_id"] = (
# #         df.Patient.astype(str) + "_" +
# #         df.cx.astype(int).astype(str) + "_" +
# #         df.cy.astype(int).astype(str)
# #     )

# #     needed = pd.read_csv(NEEDED_IDS)
# #     df = df.merge(needed, on="global_cell_id", how="inner")

# #     print(f"Using {len(df)} cells for all runs")

# #     # ---------- Model ----------
# #     model = MultiHeadResNet18().to(DEVICE).half().eval()

# #     # ---------- Runs: baseline + marker ablations ----------
# #     runs = [("baseline", -1)] + [
# #     (f"marker_{m:02d}_{MARKER_NAMES[m]}", m)
# #     for m in range(NUM_MARKERS)
# # ]
# #     summary = []

# #     for tag, m in runs:
# #         print(f"\n🧪 RUN: {tag}")

# #         out_csv = EMB_OUTDIR / f"{tag}.csv"
# #         scima_dir = SCIMA_OUTDIR / tag
# #         scima_dir.mkdir(exist_ok=True, parents=True)

# #         # ====================================================
# #         # EMBEDDING
# #         # ====================================================
# #         if not out_csv.exists():
# #             write_header = True
# #             with open(out_csv, "w") as fout:

# #                 for pid in tqdm(df.Patient.unique(), desc=f"Patients ({tag})", dynamic_ncols=True):
# #                     z = zarr.open(os.path.join(ROOT_DATA, str(pid), "data.zarr"), "r")
# #                     sdf = df[df.Patient == pid]

# #                     cell_pbar = tqdm(
# #                         total=len(sdf),
# #                         desc=f"Cells (P{pid})",
# #                         dynamic_ncols=True,
# #                         leave=False
# #                     )

# #                     for ty in range(0, z.shape[1], TILE_SIZE):
# #                         for tx in range(0, z.shape[2], TILE_SIZE):

# #                             ss = sdf[
# #                                 (sdf.cx >= tx) & (sdf.cx < tx + TILE_SIZE) &
# #                                 (sdf.cy >= ty) & (sdf.cy < ty + TILE_SIZE)
# #                             ]
# #                             if ss.empty:
# #                                 continue

# #                             tile = torch.from_numpy(
# #                                 z[:, ty:ty+TILE_SIZE, tx:tx+TILE_SIZE].astype(np.float32)
# #                             ).to(DEVICE).half()

# #                             centers = torch.stack([
# #                                 torch.tensor(ss.cx.values - tx),
# #                                 torch.tensor(ss.cy.values - ty)
# #                             ], dim=1).to(DEVICE)

# #                             for i in range(0, len(ss), GPU_BATCH_SIZE):
# #                                 sub = centers[i:i+GPU_BATCH_SIZE]
# #                                 idxs = ss.index[i:i+GPU_BATCH_SIZE]

# #                                 patches = extract_patches(tile, sub)
# #                                 patches = torch.arcsinh(patches / ARCSINH_COFACTOR)

# #                                 if m != -1:
# #                                     patches[:, m] = 0.0

# #                                 with torch.no_grad():
# #                                     emb = model(patches)

# #                                 emb = emb.float().cpu().numpy()

# #                                 for j, idx in enumerate(idxs):
# #                                     row = ss.loc[idx]
# #                                     meta = {
# #                                         "cell_index": idx,
# #                                         "Patient": row.Patient,
# #                                         "Tissue": row.Tissue,
# #                                         "Class0": row.Class0,
# #                                         "cx": row.cx,
# #                                         "cy": row.cy,
# #                                     }
# #                                     for d in range(OUT_DIM):
# #                                         meta[f"emb_{d}"] = emb[j, d]

# #                                     if write_header:
# #                                         fout.write(",".join(meta.keys()) + "\n")
# #                                         write_header = False
# #                                     fout.write(",".join(map(str, meta.values())) + "\n")

# #                                 cell_pbar.update(len(idxs))
# #                                 cell_pbar.set_postfix(
# #                                     speed=f"{cell_pbar.format_dict['rate']:.1f} cells/s"
# #                                 )

# #                             del tile
# #                             torch.cuda.empty_cache()

# #                     cell_pbar.close()

# #         # ====================================================
# #         # SCIMa (LOGGED)
# #         # ====================================================
# #         log_file = scima_dir / "log.txt"
# #         if not log_file.exists():
# #             with open(log_file, "w") as lf:
# #                 try:
# #                     subprocess.run(
# #                         [
# #                             "python", "s3_analysis.py",
# #                             "--data", str(out_csv),
# #                             "--out", str(scima_dir),
# #                             "--run_mait",
# #                             "--K", "50",
# #                             "--nrun", "10"
# #                         ],
# #                         stdout=lf,
# #                         stderr=lf,
# #                         check=True
# #                     )
# #                 except subprocess.CalledProcessError as e:
# #                     print(f"❌ SCIMa failed for {tag}, skipping")
# #                     continue

# #         # ====================================================
# #         # PARSE ACCURACY
# #         # ====================================================
# #         acc = parse_accuracy(scima_dir)
# #         print("accuracy: ",acc)
     

# #         summary.append({
# #             "run": tag,
# #             "masked_marker_index": "none" if m == -1 else m,
# #             "masked_marker_name": "none" if m == -1 else MARKER_NAMES[m],
# #             "accuracy": acc
# #         })

# #         gc.collect()

# #     # ========================================================
# #     # SAVE SUMMARY
# #     # ========================================================
# #     pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)
# #     print(f"\n📊 Summary written to {SUMMARY_CSV}")

# # # ============================================================
# # if __name__ == "__main__":
# #     main()