#!/usr/bin/env python3
# coding: utf-8

"""
Evaluation script for NativeResNet18 (38-channel)
------------------------------------------------
• Loads BEST model
• Evaluates on TEST patient (last patient)
• Uses y_center labels (same as training)
• Prints:
    - Accuracy
    - Recall per class
    - Confusion matrix
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import numpy as np
import pandas as pd
import zarr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18, ResNet18_Weights
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from tqdm import tqdm   

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "/mnt/volumec/Aswath/selected_samples.csv"
ZARR_ROOT = "/mnt/volumec/Aswath/processed_data/data"
BEST_PATH = "best_native38_multihead_scratchtrain256.pt"

PATCH_SIZE = 256
HALF = PATCH_SIZE // 2
ARCSINH_COFACTOR = 5.0

LABEL_ORDER = ["core", "normalLiver", "rim"]
LABEL_MAP = {k: i for i, k in enumerate(LABEL_ORDER)}

NUM_CLASSES = 3
NUM_INPUT_CHANNELS = 38

BATCH_SIZE = 128
NUM_WORKERS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATASET
# ============================================================
class MajorityLabelCache:
    def __init__(self, dfp):
        pts = np.vstack([dfp.cx.values, dfp.cy.values]).T
        self.tree = cKDTree(pts)
        self.labels = dfp.Tissue.values
        self.radius = HALF * np.sqrt(2)

    def majority(self, cx, cy):
        idx = self.tree.query_ball_point([cx, cy], self.radius)
        if not idx:
            return None
        vals, cnts = np.unique(self.labels[idx], return_counts=True)
        return vals[cnts.argmax()]

class PatchDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.zcache = {}
        self.mcache = {}

    def _z(self, pid):
        if pid not in self.zcache:
            self.zcache[pid] = zarr.open(
                os.path.join(ZARR_ROOT, str(pid), "data.zarr"), "r"
            )
        return self.zcache[pid]

    def _m(self, pid):
        if pid not in self.mcache:
            dfp = self.df[self.df.Patient == pid][["cx", "cy", "Tissue"]]
            self.mcache[pid] = MajorityLabelCache(dfp)
        return self.mcache[pid]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid, cx, cy = row.Patient, int(row.cx), int(row.cy)

        z = self._z(pid)
        _, H, W = z.shape

        x0, x1 = cx - HALF, cx + HALF
        y0, y1 = cy - HALF, cy + HALF

        sx0, sx1 = max(0, x0), min(W, x1)
        sy0, sy1 = max(0, y0), min(H, y1)

        patch = np.array(z[:, sy0:sy1, sx0:sx1], np.float32)

        pad = (
            (0, 0),
            (max(0, -y0), max(0, y1 - H)),
            (max(0, -x0), max(0, x1 - W)),
        )
        patch = np.pad(patch, pad, mode="reflect")

        patch = np.arcsinh(patch / ARCSINH_COFACTOR)
        x = torch.from_numpy(patch)

        y_center = LABEL_MAP[row.Tissue]
        maj = self._m(pid).majority(cx, cy)
        # 🔹 center label
        y_center = LABEL_MAP[row.Tissue]         

        return x, y_center
    
def class_accuracy(y_true, y_pred, class_idx):
    mask = (y_true == class_idx)
    if mask.sum() == 0:
        return 0.0
    return (y_pred[mask] == y_true[mask]).mean()

# ============================================================
# MODEL
# ============================================================
class NativeResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.norm = nn.InstanceNorm2d(NUM_INPUT_CHANNELS, affine=True)

        old = base.conv1
        self.first_conv = nn.Conv2d(
            NUM_INPUT_CHANNELS, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

        with torch.no_grad():
            w = old.weight.mean(dim=1, keepdim=True)
            self.first_conv.weight.copy_(w.repeat(1, NUM_INPUT_CHANNELS, 1, 1))

        self.stem = nn.Sequential(
            self.first_conv,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.backbone = nn.Sequential(
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4
        )

        d = base.fc.in_features
        self.head_major = nn.Linear(d, NUM_CLASSES)
        self.head_center = nn.Linear(d, NUM_CLASSES)

    def forward(self, x):
        x = self.norm(x)
        x = self.stem(x)
        x = self.backbone(x)
        x = x.mean(dim=[2,3])
        return self.head_major(x), self.head_center(x)

class NativeResNet18scratch(nn.Module):
    def __init__(self):
        super().__init__()

        #  NO ImageNet pretraining
        base = resnet18(weights=None)
        
        # -------- Instance Normalization for 38-channel input --------
        self.norm = nn.InstanceNorm2d(
            NUM_INPUT_CHANNELS,
            affine=True,
            track_running_stats=False
        )


        # -------- First conv: 38-channel input --------
        old = base.conv1
        self.first_conv = nn.Conv2d(
            in_channels=NUM_INPUT_CHANNELS,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

        #  Proper initialization for scratch training
        nn.init.kaiming_normal_(
            self.first_conv.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

        # -------- Stem --------
        self.stem = nn.Sequential(
            self.first_conv,
            base.bn1,
            base.relu,
            base.maxpool
        )

        # -------- Backbone --------
        self.backbone = nn.Sequential(
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4
        )

        # -------- Heads --------
        d = base.fc.in_features
        self.head_major = nn.Linear(d, NUM_CLASSES)
        self.head_center = nn.Linear(d, NUM_CLASSES)

    def forward(self, x):
        x = self.norm(x)  
        x = self.stem(x)
        x = self.backbone(x)
        x = x.mean(dim=(2, 3))   # global average pooling
        return self.head_major(x), self.head_center(x)

    
# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_csv(CSV_PATH)
    df["cx"] = (df.XMin + df.XMax) / 2
    df["cy"] = (df.YMin + df.YMax) / 2

    test_pid = sorted(df.Patient.unique())[-1]
    test_df = df[df.Patient == test_pid]

    test_ds = PatchDataset(test_df)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # model = NativeResNet18().to(DEVICE)
    model = NativeResNet18scratch().to(DEVICE)
    state = torch.load(BEST_PATH, map_location=DEVICE,  weights_only=False )
    model.load_state_dict(state, strict=False)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating test set"):
            x = x.to(DEVICE)
            logits_major, logits_center = model(x)
            preds = logits_center.argmax(1).cpu().numpy()
            # preds = logits_center.argmax(1)
            # preds = logits.argmax(1).cpu().numpy()

            y_pred.append(preds)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

    print("\n=== TEST EVALUATION (y_center) ===")
    print(f"Accuracy: {acc:.4f}\n")

    print("Recall per class:")
    for i, name in enumerate(LABEL_ORDER):
        print(f"  {name:12s}: {rec[i]:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    
        
    acc_core = class_accuracy(y_true, y_pred, LABEL_MAP["core"])
    acc_rim  = class_accuracy(y_true, y_pred, LABEL_MAP["rim"])


if __name__ == "__main__":
    main() 