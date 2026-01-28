#!/usr/bin/env python3
# coding: utf-8

import os, random
import numpy as np
import pandas as pd
import zarr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from scipy.spatial import cKDTree
from tqdm import tqdm

# ================= CONFIG =================
CSV_PATH = "/mnt/volumec/Aswath/selected_samples.csv"
ZARR_ROOT = "/mnt/volumec/Aswath/processed_data/data"

PATCH_SIZE = 128
HALF = PATCH_SIZE // 2
ARCSINH_COFACTOR = 5.0


LABEL_ORDER = ["core", "normalLiver", "rim"]
LABEL_MAP = {k: i for i, k in enumerate(LABEL_ORDER)}
NUM_CLASSES = 3
NUM_INPUT_CHANNELS = 38
NUM_WORKERS = 8

EPOCHS = 5
BATCH_SIZE = 1024
AUX_W = 0.3
LR_BACKBONE = 1e-4
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
LR_ADAPTER  = 1e-3
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EVAL_BS  = 256

BEST_PATH = "best_native38_multihead_linear.pt"

# ================= UTILS =================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def recall_dict(y, p):
    r = recall_score(y, p, labels=[0,1,2], average=None, zero_division=0)
    return dict(zip(LABEL_ORDER, r))

# ================= DATA =================
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
            dfp = self.df[self.df.Patient == pid][["cx","cy","Tissue"]]
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

        sx0, sx1 = max(0,x0), min(W,x1)
        sy0, sy1 = max(0,y0), min(H,y1)

        patch = np.array(z[:, sy0:sy1, sx0:sx1], np.float32)

        pad = (
            (0,0),
            (max(0,-y0), max(0,y1-H)),
            (max(0,-x0), max(0,x1-W))
        )
        patch = np.pad(patch, pad, mode="reflect")

        patch = np.arcsinh(patch / ARCSINH_COFACTOR)
        x = torch.from_numpy(patch)

        y_center = LABEL_MAP[row.Tissue]
        maj = self._m(pid).majority(cx, cy)
        y_major = LABEL_MAP[maj] if maj else y_center

        return x, y_major, y_center

def collate(batch):
    x, ym, yc = zip(*batch)
    return torch.stack(x), torch.tensor(ym), torch.tensor(yc)

# ================= MODEL =================
class NativeResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.norm = nn.InstanceNorm2d(NUM_INPUT_CHANNELS, affine=True)

        old = base.conv1
         # ✅ PURE LINEAR PROJECTION (38 → 3)
        self.adapter = nn.Conv2d(
            NUM_INPUT_CHANNELS,
            3,
            kernel_size=1,
            bias=False
        )

       # ImageNet backbone (RGB-compatible now)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        d = base.fc.in_features

        # Multi-head outputs
        self.head_major = nn.Linear(d, NUM_CLASSES)
        self.head_center = nn.Linear(d, NUM_CLASSES)

    def forward(self, x):
        # x: (B, 38, H, W)
        x = self.adapter(x)          # (B, 3, H, W) ← linear mix
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head_major(x), self.head_center(x)
    
    
class NativeResNet18scratch(nn.Module):
    def __init__(self):
        super().__init__()

        # NO ImageNet pretraining
        base = resnet18(weights=None)

        #  Disable inplace ReLU (important for explainability & hooks)
        for m in base.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
                
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

        # ✅ Proper initialization for scratch training
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

# ================= TRAIN =================
def main():
    seed_everything(SEED)

    df = pd.read_csv(CSV_PATH)
    df["cx"] = (df.XMin + df.XMax) / 2
    df["cy"] = (df.YMin + df.YMax) / 2

    patients = sorted(df.Patient.unique())
    test_pid = patients[-1]

    df_tv = df[df.Patient != test_pid]
    df_test = df[df.Patient == test_pid]

    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
    tr_i, va_i = next(sss.split(df_tv, df_tv.Tissue))
    tr_df, va_df = df_tv.iloc[tr_i], df_tv.iloc[va_i]

    loaders = {
        "train": DataLoader(PatchDataset(tr_df), BATCH_SIZE, True, collate_fn=collate,num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True),
        "val":   DataLoader(PatchDataset(va_df), EVAL_BS, False, collate_fn=collate,num_workers=0,pin_memory=False,
persistent_workers=False),
        "test":  DataLoader(PatchDataset(df_test), EVAL_BS, False, collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=False,
persistent_workers=False),
    }

    # model = NativeResNet18().to(DEVICE)
    model = NativeResNet18().to(DEVICE)

    # class weights (FAST)
    y = tr_df.Tissue.map(LABEL_MAP).values
    w = np.bincount(y, minlength=3)
    w = torch.tensor((w.sum()/(w+1e-6))/np.mean(w.sum()/(w+1e-6)),
                     device=DEVICE,dtype=torch.float32)

    crit_major = nn.CrossEntropyLoss(weight=w)
    crit_center = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(
    [
        # 🔹 Linear adapter (38 → 3) — learn fast
        {"params": model.adapter.parameters(), "lr": LR_ADAPTER},

        # 🔹 ImageNet backbone — learn slowly
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},

        # 🔹 Classification heads — learn fast
        {"params": model.head_major.parameters(), "lr": LR_HEAD},
        {"params": model.head_center.parameters(), "lr": LR_HEAD},
    ],
    weight_decay=WEIGHT_DECAY
)

    best = -1.0

    for ep in range(EPOCHS):
        model.train()
        for x, ym, yc in tqdm(loaders["train"], desc=f"TRAIN {ep+1}"):
            x, ym, yc = x.to(DEVICE), ym.to(DEVICE), yc.to(DEVICE)
            opt.zero_grad()
            om, oc = model(x)
            loss = crit_major(om, ym) + AUX_W * crit_center(oc, yc)
            loss.backward()
            opt.step()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y, _ in tqdm(loaders["val"], desc="VAL"):
                p = model(x.to(DEVICE))[0].argmax(1).cpu().numpy()
                yt.append(y.numpy()); yp.append(p)
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        mean_recall = np.mean(list(recall_dict(yt, yp).values()))

        if mean_recall > best:
            best = mean_recall
            torch.save(model.state_dict(), BEST_PATH)
            print(f" Saved best model ({best:.3f})")

    print("Training complete.")
    