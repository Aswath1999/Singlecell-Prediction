#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

import modules_scima as ms
from utils import mkdir_p

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
np.random.seed(12345)

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run S3-CIMA on CNN embeddings")

    # parser.add_argument(
    #     "--data",
    #     default="/mnt/volumec/Aswath/patchmodel/s3cima/marker_embeddings128_contrastive_Tissue.csv",
    #     help="Path to CSV file with embeddings",
    # )
    # parser.add_argument(
    #     "--data",
    #     default="/mnt/volumec/Aswath/patchmodel/s3cima/CNNResnetembeddings/arcsin_varpatch_resnet18.csv",
    #     help="Path to CSV file with embeddings",
    # )
    
    parser.add_argument(
        "--data",
        default="/mnt/volumec/Aswath/patchmodel/s3cima/CNNResnetembeddings/38chresnetscratch.csv",
        help="Path to CSV file with embeddings",
    )
    parser.add_argument(
        "--out",
        default='/mnt/volumec/Aswath/patchmodel/s3cima/s3cimares/markeraware/batch1024/resultsMAIT',
        help="Output directory for SCIMa results",
    )

    parser.add_argument("--run_bg", action="store_true", help="Run BG anchor analysis")
    parser.add_argument("--run_mait", action="store_true", help="Run MAIT anchor analysis")

    parser.add_argument("--K", type=int, default=50, help="Number of neighbors")
    parser.add_argument("--N", type=int, default=500, help="Number of random sets")
    parser.add_argument("--nrun", type=int, default=10, help="Number of SCIMa runs")

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()

    logging.info("Loading data: %s", args.data)
    
    Data = pd.read_csv(args.data)
    # Data["cx"] = (Data.XMin + Data.XMax) / 2
    # Data["cy"] = (Data.YMin + Data.YMax) / 2

    # --------------------------------------------------
    # Rename columns
    # --------------------------------------------------
    Data = Data.rename(columns={
        "cx": "spatial_dim0",
        "cy": "spatial_dim1",
        "Class0": "cell_type",
        "Tissue": "condition",
        "Patient": "sample",
    })
    
    # Data["cell_id"] = Data.index
    # cellid = Data["cell_id"].to_numpy() 
    Data["global_cell_id"] = (
    Data["sample"].astype(str) + "_" +
    Data["spatial_dim0"].astype(int).astype(str) + "_" +
    Data["spatial_dim1"].astype(int).astype(str)
)

    cellid = Data["global_cell_id"].to_numpy()

    # --------------------------------------------------
    # Metadata & marker columns
    # --------------------------------------------------
    meta_cols = [
        "cell_index",
        "sample",
        "spatial_dim0",
        "spatial_dim1",
        "condition",
        "cell_type",
        "patch_size",
    ]

    # marker_cols = [c for c in Data.columns if c not in meta_cols]

    marker_cols = [c for c in Data.columns if c.startswith('emb_')]

    logging.info("Number of marker columns: %d", len(marker_cols))

    Intensity = Data[marker_cols].astype(float).values

    # --------------------------------------------------
    # IDs
    # --------------------------------------------------
    # Data["cell_id"] = Data.index
    mappings = {k: i for i, k in enumerate(Data["sample"].unique())}
    Data["image_id"] = Data["sample"].map(mappings)

    # --------------------------------------------------
    # Arrays for SCIMa
    # --------------------------------------------------
    pat = Data["sample"].to_numpy()
    image = Data["image_id"].to_numpy()
    ct = Data["cell_type"].to_numpy()
    x = Data["spatial_dim0"].to_numpy()
    y = Data["spatial_dim1"].to_numpy()
    # cellid = Data["cell_id"].to_numpy()
    groups = Data["condition"].to_numpy()

    labels = np.unique(groups)

    mkdir_p(args.out)

    # ==================================================
    # BG ANCHOR
    # ==================================================
    if args.run_bg:
        Anchor = "BG"
        K = args.K
        OUTDIR = os.path.join(args.out, f"Anchor{Anchor}_K{K}")
        mkdir_p(OUTDIR)

        logging.info("Running BG spatial input generation")

        for lab in labels:
            inx = np.where(groups == lab)[0]
            ms.BG_spatial_input_per_sample(
                args.N,
                image[inx],
                pat[inx],
                Intensity[inx, :],
                ct[inx],
                x[inx],
                y[inx],
                cellid[inx],
                K,
                lab,
                OUTDIR,
            )

        logging.info("Running BG SCIMa")
        ms.run_scima(
            Anchor=Anchor,
            ntrain_per_class=2,
            K=K,
            k=K,
            nset_thr=1,
            labels=["core", "normalLiver", "rim"],
            classes=[0, 1, 2],
            path=args.out,
            nrun=args.nrun,
            background=False,
        )

    # ==================================================
    # MAIT ANCHOR
    # ==================================================
    if args.run_mait:
        Anchor = "MAITs"
        K = args.K
        OUTDIR = os.path.join(args.out, f"Anchor{Anchor}_K{K}")
        mkdir_p(OUTDIR)

        logging.info("Running MAIT spatial input generation")

        for lab in labels:
            inx = np.where(groups == lab)[0]
            ms.spatial_input_per_sample(
                Anchor,
                image[inx],
                pat[inx],
                Intensity[inx, :],
                ct[inx],
                x[inx],
                y[inx],
                cellid[inx],
                K,
                lab,
                OUTDIR,
            )

        logging.info("Running MAIT SCIMa (with background)")
        ms.run_scima(
            Anchor=Anchor,
            ntrain_per_class=2,
            K=K,
            k=10,
            nset_thr=0.9,
            labels=[
                "core",
                "normalLiver",
                "rim",
                "randcore",
                "randnormalLiver",
                "randrim",
            ],
            classes=[0, 1, 2, 0, 1, 2],
            path=args.out,
            nrun=args.nrun,
            background=True,
        )

    logging.info("SCIMa pipeline finished successfully")
    
    

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()