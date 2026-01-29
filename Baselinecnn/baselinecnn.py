#!/usr/bin/env python3
# coding: utf-8

import os, json, random, time
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import zarr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==========================================================
# ======================== CONFIG ==========================
# ==========================================================
CSV_PATH   = "/mnt/volumec/Aswath/selected_samples.csv"
ZARR_ROOT  = "/mnt/volumec/Aswath/processed_data/data"

ARCSINH_COFACTOR = 5.0
LABEL_ORDER = ["core", "normalLiver", "rim"]
LABEL_MAP = {k: i for i, k in enumerate(LABEL_ORDER)}

NUM_CLASSES = 3
NUM_INPUT_CHANNELS = 38

PATCH_SIZES = [32, 64, 128, 256, 512]

EPOCHS_SWEEP = 5
EPOCHS_MULTI = 5

LR_GRID = [1e-3, 5e-4, 2e-4]
DROPOUT_GRID = [0.0, 0.1, 0.2]

WEIGHT_DECAY = 1e-4
AUX_W = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

OUTDIR = "smallcnn_v100_safe_logs"
os.makedirs(OUTDIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTDIR, "baseline_sweep_results.csv")
BEST_BASELINE_JSON = os.path.join(OUTDIR, "best_baseline_config.json")
BEST_BASELINE_CKPT = os.path.join(OUTDIR, "best_smallcnn_baseline.pt")
BEST_MULTI_CKPT = os.path.join(OUTDIR, "best_smallcnn_multihead.pt")

torch.backends.cudnn.benchmark = True

# ==========================================================
# ======================== UTILS ===========================
# ==========================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def recall_dict(y, p):
    r = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    return dict(zip(LABEL_ORDER, r))

def mean_recall(y, p):
    d = recall_dict(y, p)
    return float(np.mean(list(d.values()))), d

def append_csv(path, row):
    df = pd.DataFrame([row])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ==========================================================
# ======= MEMORY-SAFE LOADER PLAN (V100 32 GB) ==============
# ==========================================================
def loader_plan(patch):
    """
    Conservative, safe settings for 38-channel patches
    on V100 32 GB + 16 CPU cores.
    """
    if patch <= 64:
        return dict(bs=1024, workers=12, prefetch=2, pin=True)
    if patch == 128:
        return dict(bs=512, workers=10, prefetch=2, pin=True)
    if patch == 256:
        return dict(bs=256, workers=8, prefetch=1, pin=True)
    if patch == 512:
        return dict(bs=16, workers=4, prefetch=1, pin=True)
    raise ValueError("Unsupported patch size")

# ==========================================================
# ======================== DATA ============================
# ==========================================================
class MajorityLabelCache:
    def __init__(self, df_patient, radius):
        pts = np.vstack([df_patient.cx, df_patient.cy]).T
        self.tree = cKDTree(pts)
        self.labels = df_patient.Tissue.values
        self.radius = radius

    def majority(self, cx, cy):
        idx = self.tree.query_ball_point([cx, cy], self.radius)
        if len(idx) == 0:
            return None
        vals, cnts = np.unique(self.labels[idx], return_counts=True)
        return vals[cnts.argmax()]

class PatchDataset(Dataset):
    def __init__(self, df_split, df_all, patch_size):
        self.df = df_split.reset_index(drop=True)
        self.df_all = df_all
        self.ps = patch_size
        self.half = patch_size // 2
        self.radius = self.half * np.sqrt(2)
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
            self.mcache[pid] = MajorityLabelCache(dfp, self.radius)
        return self.mcache[pid]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid, cx, cy = row.Patient, int(row.cx), int(row.cy)

        z = self._z(pid)
        _, H, W = z.shape

        x0, x1 = cx - self.half, cx + self.half
        y0, y1 = cy - self.half, cy + self.half

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

# ==========================================================
# ======================== MODELS ==========================
# ==========================================================
class SmallCNN(nn.Module):
    def __init__(self, C, n_classes, dropout=0.1, use_in=False):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(C, affine=True) if use_in else None

        self.features = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        if self.inorm is not None:
            x = self.inorm(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(self.drop(x))

class SmallCNNMultiHead(nn.Module):
    def __init__(self, C, n_classes, dropout=0.1):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(C, affine=True)

        self.features = SmallCNN(C, n_classes, dropout, use_in=False).features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)

        self.head_major = nn.Linear(128, n_classes)
        self.head_center = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.inorm(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.head_major(x), self.head_center(x)

# ==========================================================
# ====================== TRAINING ==========================
# ==========================================================
@dataclass
class RunConfig:
    run_type: str
    patch_size: int
    use_in: bool
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    workers: int
    prefetch: int
    seed: int = SEED

def make_loaders(df, patch):
    plan = loader_plan(patch)

    patients = sorted(df.Patient.unique())
    test_pid = patients[-1]

    df_tv = df[df.Patient != test_pid]
    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
    tr_i, va_i = next(sss.split(df_tv, df_tv.Tissue))
    tr_df, va_df = df_tv.iloc[tr_i], df_tv.iloc[va_i]

    train_loader = DataLoader(
        PatchDataset(tr_df, df, patch),
        batch_size=plan["bs"],
        shuffle=True,
        num_workers=plan["workers"],
        prefetch_factor=plan["prefetch"],
        pin_memory=plan["pin"],
        persistent_workers=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        PatchDataset(va_df, df, patch),
        batch_size=plan["bs"] // 2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    return train_loader, val_loader, tr_df, plan

# ==========================================================
# ======================== MAIN ============================
# ==========================================================
def main():
    seed_everything(SEED)

    df = pd.read_csv(CSV_PATH)
    df["cx"] = (df.XMin + df.XMax) / 2
    df["cy"] = (df.YMin + df.YMax) / 2

    best_row, best_state = None, None

    print(f"[{now()}] Starting baseline sweep")

    for ps in PATCH_SIZES:
        for use_in in [False, True]:
            for lr in LR_GRID:
                for do in DROPOUT_GRID:

                    train_loader, val_loader, tr_df, plan = make_loaders(df, ps)

                    model = SmallCNN(
                        NUM_INPUT_CHANNELS, NUM_CLASSES,
                        dropout=do, use_in=use_in
                    ).to(DEVICE)

                    y = tr_df.Tissue.map(LABEL_MAP).values
                    w = np.bincount(y, minlength=NUM_CLASSES)
                    w = torch.tensor(
                        (w.sum() / (w + 1e-6)) / np.mean(w.sum() / (w + 1e-6)),
                        device=DEVICE, dtype=torch.float32
                    )

                    crit = nn.CrossEntropyLoss(weight=w)
                    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

                    best_mr = -1.0
                    for ep in range(EPOCHS_SWEEP):
                        model.train()
                        for x, y, _ in train_loader:
                            x, y = x.to(DEVICE), y.to(DEVICE)
                            opt.zero_grad(set_to_none=True)
                            loss = crit(model(x), y)
                            loss.backward()
                            opt.step()

                        model.eval()
                        yt, yp = [], []
                        with torch.no_grad():
                            for x, y, _ in val_loader:
                                p = model(x.to(DEVICE)).argmax(1).cpu().numpy()
                                yt.append(y.numpy())
                                yp.append(p)
                        yt = np.concatenate(yt)
                        yp = np.concatenate(yp)
                        mr, per_cls = mean_recall(yt, yp)

                        if mr > best_mr:
                            best_mr = mr
                            best_state_local = model.state_dict()

                    row = {
                        "timestamp": now(),
                        "patch_size": ps,
                        "use_instance_norm": use_in,
                        "lr": lr,
                        "dropout": do,
                        "batch_size": plan["bs"],
                        "workers": plan["workers"],
                        "prefetch": plan["prefetch"],
                        "val_mean_recall": best_mr,
                        **{f"recall_{k}": v for k, v in per_cls.items()}
                    }
                    append_csv(RESULTS_CSV, row)

                    print(f"ps={ps} IN={use_in} lr={lr} do={do} -> mr={best_mr:.4f}")

                    if best_row is None or best_mr > best_row["val_mean_recall"]:
                        best_row = row
                        best_state = best_state_local
                        torch.save(best_state, BEST_BASELINE_CKPT)
                        with open(BEST_BASELINE_JSON, "w") as f:
                            json.dump(best_row, f, indent=2)

    print("\nBest baseline:")
    print(json.dumps(best_row, indent=2))

    # ================= MULTIHEAD =================
    print(f"\n[{now()}] Training multi-head model")

    ps = best_row["patch_size"]
    train_loader, val_loader, tr_df, plan = make_loaders(df, ps)

    model = SmallCNNMultiHead(NUM_INPUT_CHANNELS, NUM_CLASSES, dropout=best_row["dropout"]).to(DEVICE)

    y = tr_df.Tissue.map(LABEL_MAP).values
    w = np.bincount(y, minlength=NUM_CLASSES)
    w = torch.tensor(
        (w.sum() / (w + 1e-6)) / np.mean(w.sum() / (w + 1e-6)),
        device=DEVICE, dtype=torch.float32
    )

    crit = nn.CrossEntropyLoss(weight=w)
    opt = optim.AdamW(model.parameters(), lr=best_row["lr"], weight_decay=WEIGHT_DECAY)

    best_mr = -1.0
    for ep in range(EPOCHS_MULTI):
        model.train()
        for x, ym, yc in train_loader:
            x, ym, yc = x.to(DEVICE), ym.to(DEVICE), yc.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            om, oc = model(x)
            loss = crit(om, ym) + AUX_W * crit(oc, yc)
            loss.backward()
            opt.step()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                p = model(x.to(DEVICE))[0].argmax(1).cpu().numpy()
                yt.append(y.numpy())
                yp.append(p)
        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        mr, _ = mean_recall(yt, yp)

        if mr > best_mr:
            best_mr = mr
            torch.save(model.state_dict(), BEST_MULTI_CKPT)

    print(f"✅ Multi-head best mean recall = {best_mr:.4f}")
    print("All results logged to:", RESULTS_CSV)

if __name__ == "__main__":
    main()