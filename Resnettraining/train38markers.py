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

from torchvision.models import resnet18
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score
from scipy.spatial import cKDTree
from tqdm import tqdm

# ================= CONFIG =================
CSV_PATH = "/mnt/volumec/Aswath/selected_samples.csv"
ZARR_ROOT = "/mnt/volumec/Aswath/processed_data/data"

PATCH_SIZE = 256
HALF = PATCH_SIZE // 2
ARCSINH_COFACTOR = 5.0

LABEL_ORDER = ["core", "normalLiver", "rim"]
LABEL_MAP = {k: i for i, k in enumerate(LABEL_ORDER)}
NUM_CLASSES = 3
NUM_INPUT_CHANNELS = 38

NUM_WORKERS = 8
EPOCHS = 5
BATCH_SIZE = 512
EVAL_BS = 256

AUX_W = 0.3
LR_BACKBONE = 1e-4
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BEST_PATH = "best_native38_multihead_scratchtraining256.pt"

# ================= UTILS =================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def recall_dict(y, p):
    r = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    return dict(zip(LABEL_ORDER, r))

# ================= DATA =================
class MajorityLabelCache:
    """KDTree over ALL annotations of a patient (correct!)"""
    def __init__(self, df_patient):
        pts = np.vstack([df_patient.cx.values, df_patient.cy.values]).T
        self.tree = cKDTree(pts)
        self.labels = df_patient.Tissue.values
        self.radius = HALF * np.sqrt(2)

    def majority(self, cx, cy):
        idx = self.tree.query_ball_point([cx, cy], self.radius)
        if len(idx) == 0:
            return None
        vals, cnts = np.unique(self.labels[idx], return_counts=True)
        return vals[cnts.argmax()]

class PatchDataset(Dataset):
    def __init__(self, df_split, df_all):
        self.df = df_split.reset_index(drop=True)
        self.df_all = df_all
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
            dfp = self.df_all[self.df_all.Patient == pid][["cx", "cy", "Tissue"]]
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
        y_major = LABEL_MAP.get(maj, y_center)

        return x, y_major, y_center

def collate(batch):
    x, ym, yc = zip(*batch)
    return torch.stack(x), torch.tensor(ym), torch.tensor(yc)

# ================= MODEL =================
class NativeResNet18Scratch(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)

        for m in base.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        self.norm = nn.InstanceNorm2d(
            NUM_INPUT_CHANNELS, affine=True, track_running_stats=False
        )

        old = base.conv1
        self.first_conv = nn.Conv2d(
            NUM_INPUT_CHANNELS,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.first_conv.weight, mode="fan_out", nonlinearity="relu"
        )

        self.stem = nn.Sequential(
            self.first_conv,
            base.bn1,
            base.relu,
            base.maxpool,
        )

        self.backbone = nn.Sequential(
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        d = base.fc.in_features
        self.head_major = nn.Linear(d, NUM_CLASSES)
        self.head_center = nn.Linear(d, NUM_CLASSES)

    def forward(self, x):
        x = self.norm(x)
        x = self.stem(x)
        x = self.backbone(x)
        x = x.mean(dim=(2, 3))
        return self.head_major(x), self.head_center(x)

# ================= TRAIN =================
def main():
    seed_everything(SEED)

    df = pd.read_csv(CSV_PATH)
    df["cx"] = (df.XMin + df.XMax) / 2
    df["cy"] = (df.YMin + df.YMax) / 2

    patients = sorted(df.Patient.unique())
    test_pid = patients[-1]  # explicit holdout

    df_tv = df[df.Patient != test_pid]
    df_test = df[df.Patient == test_pid]

    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
    tr_i, va_i = next(sss.split(df_tv, df_tv.Tissue))
    tr_df, va_df = df_tv.iloc[tr_i], df_tv.iloc[va_i]

    loaders = {
        "train": DataLoader(
            PatchDataset(tr_df, df),
            BATCH_SIZE,
            shuffle=True,
            collate_fn=collate,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        ),
        "val": DataLoader(
            PatchDataset(va_df, df),
            EVAL_BS,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        ),
    }

    model = NativeResNet18Scratch().to(DEVICE)

    y = tr_df.Tissue.map(LABEL_MAP).values
    w = np.bincount(y, minlength=NUM_CLASSES)
    w = torch.tensor(
        (w.sum() / (w + 1e-6)) / np.mean(w.sum() / (w + 1e-6)),
        device=DEVICE,
        dtype=torch.float32,
    )

    crit_major = nn.CrossEntropyLoss(weight=w)
    crit_center = nn.CrossEntropyLoss(weight=w)

    opt = optim.AdamW(
        [
            {"params": model.first_conv.parameters(), "lr": LR_BACKBONE},
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.head_major.parameters(), "lr": LR_HEAD},
            {"params": model.head_center.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
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
                yt.append(y.numpy())
                yp.append(p)

        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        mean_recall = np.mean(list(recall_dict(yt, yp).values()))

        if mean_recall > best:
            best = mean_recall
            torch.save(model.state_dict(), BEST_PATH)
            print(f"✅ Saved best model ({best:.3f})")

    print("Training complete.")

if __name__ == "__main__":
    main()

