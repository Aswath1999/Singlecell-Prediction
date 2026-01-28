import os
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.models as models
import torch.nn as nn


import zarr
import gc

# ============================================================
# CONFIG (V100 SAFE)
# ============================================================
PATCH_SIZE = 128
HALF = PATCH_SIZE // 2
TILE_SIZE = 2048
# GPU_BATCH_SIZE = 2048
GPU_BATCH_SIZE = 1024
# GPU_BATCH_SIZE = 64

ARCSINH_COFACTOR = 5.0
OUT_DIM = 512
# OUT_DIM = 2048 #for resnet50

DEVICE = torch.device("cuda")

NUM_CLASSES = 3

NUM_INPUT_CHANNELS = 38

NUM_CLASSES = 3



# ============================================================
# UPDATED CELL CNN (STRONG BUT FAST)
# ============================================================
class CellCNN(nn.Module):
    """
    Lightweight CNN for offline embeddings.
    Optimized for speed + representation quality.
    """
    def __init__(self, in_ch=38, out_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.encoder(x).flatten(1)
        return self.fc(x)
    
    


class ConvNeXtEmbeddingNet(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()

        # Marker → RGB adapter (KEEP THIS)
        self.adapter = nn.Sequential(
            nn.Conv2d(38, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        backbone = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )

        # ❄️ Freeze everything (IMPORTANT for embeddings)
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.out_dim = out_dim

    def forward(self, x):
        x = self.adapter(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)   # (N, 1024)
        return x
    
 
class NativeResNet18(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Disable inplace ReLU (important for hooks / Grad-CAM / IG)
        for m in base.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        self.norm = nn.InstanceNorm2d(
            NUM_INPUT_CHANNELS,
            affine=True,
            track_running_stats=False
        )

        # -------- First conv: 38 → 64 (ImageNet init) --------
        old = base.conv1
        self.first_conv = nn.Conv2d(
            in_channels=NUM_INPUT_CHANNELS,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

        # Initialize from pretrained RGB conv
        with torch.no_grad():
            w = old.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
            self.first_conv.weight.copy_(
                w.repeat(1, NUM_INPUT_CHANNELS, 1, 1)
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

        self.feat_dim = base.fc.in_features  # 512

        # 🔹 Embedding head (identity by default)
        self.embed = nn.Identity()  # or nn.Linear(512, emb_dim)

        # Classification heads
        self.head_major = nn.Linear(self.feat_dim, NUM_CLASSES)
        self.head_center = nn.Linear(self.feat_dim, NUM_CLASSES)

    def forward(self, x, return_embedding=False):
        x = self.norm(x)
        x = self.stem(x)
        x = self.backbone(x)
        x = x.mean(dim=(2, 3))   # GAP → (B, 512)

        emb = self.embed(x)

        if return_embedding:
            return emb

        return self.head_major(emb), self.head_center(emb)

class NativeResNet18scratch(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()

        base = resnet18(weights=None)
        # base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


        # Disable inplace ReLU (safe for hooks / explainability)
        for m in base.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        self.norm = nn.InstanceNorm2d(
            38,
            affine=True,
            track_running_stats=False
        )

        # -------- First conv: 38 → 64 --------
        old = base.conv1
        self.first_conv = nn.Conv2d(
            in_channels=NUM_INPUT_CHANNELS,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

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

        self.feat_dim = base.fc.in_features  # 512

        # 🔹 Projection head for embeddings (optional but recommended)
        self.embed = nn.Identity()  # or nn.Linear(512, emb_dim)

        # Heads (ignored during embedding extraction)
        self.head_major = nn.Linear(self.feat_dim, NUM_CLASSES)
        self.head_center = nn.Linear(self.feat_dim, NUM_CLASSES)

    def forward(self, x, return_embedding=False):
        x = self.norm(x)
        x = self.stem(x)
        x = self.backbone(x)
        x = x.mean(dim=(2, 3))  # GAP → (B, 512)

        emb = self.embed(x)

        if return_embedding:
            return emb

        return self.head_major(emb), self.head_center(emb)
    
class MultiHeadResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        from torchvision.models import resnet50, ResNet50_Weights
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.adapter = MarkerAwareAdapter()

        #  Freeze early layers (safe + fast)
        for layer in [base.conv1, base.bn1, base.layer1]:
            for p in layer.parameters():
                p.requires_grad = False

        # Feature extractor (remove FC)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.feat_dim = base.fc.in_features  # ✅ 2048

        # Heads NOT used during embedding extraction
        self.head_major = nn.Linear(self.feat_dim, num_classes)
        self.head_center = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.adapter(x)              # (B,3,H,W)
        x = self.features(x)             # (B,2048,1,1)
        emb = torch.flatten(x, 1)        # (B,2048)

        if return_embedding:
            return emb

        return self.head_major(emb), self.head_center(emb)
    
    
@torch.no_grad()
def minmax_norm_patches(patches, eps=1e-6):
    """
    patches: (B, C, H, W)
    Normalizes each patch independently to [0, 1] per channel.
    """
    B, C, H, W = patches.shape
    patches = patches.view(B, C, -1)

    minv = patches.min(dim=2, keepdim=True)[0]
    maxv = patches.max(dim=2, keepdim=True)[0]

    patches = (patches - minv) / (maxv - minv + eps)
    return patches.view(B, C, H, W)

class MultiHeadResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # self.adapter = MarkerAwareAdapter()
#         self.adapter = nn.Sequential(
#     nn.Conv2d(38, 64, 1),
#     nn.BatchNorm2d(64),
#     nn.GELU(),
#     nn.Conv2d(64, 16, 1),
#     nn.BatchNorm2d(16),
#     nn.GELU(),
#     nn.Conv2d(16, 3, 1),
# )     
        self.adapter = nn.Sequential(
            nn.Conv2d(38, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
        )
        

        # --------------------------------------------------
        #  Freeze early ResNet layers
        # --------------------------------------------------
        for layer in [base.conv1, base.bn1, base.layer1]:
            for p in layer.parameters():
                p.requires_grad = False

        # Feature extractor (remove FC)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.feat_dim = base.fc.in_features  # 512

        # Two heads (not used during embedding extraction)
        self.head_major = nn.Linear(self.feat_dim, num_classes)
        self.head_center = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_embedding=False):
        # x: (B, 38, H, W)
        x = self.adapter(x)         # → (B, 3, H, W)
        x = self.features(x)        # → (B, 512, 1, 1)
        emb = torch.flatten(x, 1)   # → (B, 512)

        if return_embedding:
            return emb

        return self.head_major(emb), self.head_center(emb)

# ============================================================
# FAST + NUMERICALLY SAFE PATCH EXTRACTION
# ============================================================
@torch.no_grad()
def extract_patches_fast(tile, centers):
    """
    tile:    (C, H, W) float16 on GPU
    centers: (N, 2) pixel coords (cx, cy)
    returns: (N, C, PATCH_SIZE, PATCH_SIZE)
    """
    _, H, W = tile.shape
    device = tile.device

    centers = centers.float()

    cx = centers[:, 0] / (W - 1) * 2 - 1
    cy = centers[:, 1] / (H - 1) * 2 - 1

    lin = torch.linspace(
        -HALF, HALF - 1,
        PATCH_SIZE,
        device=device,
        dtype=torch.float32
    )

    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    gx = gx / (W - 1) * 2
    gy = gy / (H - 1) * 2

    grid = torch.stack([gx, gy], dim=-1)
    grid = grid.unsqueeze(0)
    grid = grid + torch.stack([cx, cy], dim=1)[:, None, None, :]
    grid = grid.clamp(-1, 1).to(tile.dtype)

    tile = tile.unsqueeze(0).expand(centers.shape[0], -1, -1, -1)

    patches = F.grid_sample(
        tile,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    return torch.nan_to_num(patches)


@torch.no_grad()
def robust_minmax_tile(tile, p_low=1.0, p_high=99.0):
    """
    tile: (C, H, W) float16 tensor on GPU
    returns: normalized tile in [0,1], float16
    """
    # quantile needs float32
    tile_f = tile.float()

    C = tile_f.shape[0]
    tile_flat = tile_f.view(C, -1)

    lo = torch.quantile(tile_flat, p_low / 100.0, dim=1, keepdim=True)
    hi = torch.quantile(tile_flat, p_high / 100.0, dim=1, keepdim=True)

    tile_flat = torch.clamp(tile_flat, lo, hi)
    tile_flat = (tile_flat - lo) / (hi - lo + 1e-6)

    # back to float16 for speed + memory
    return tile_flat.view_as(tile_f).half()


@torch.no_grad()
def fast_norm(patches):
    minv = patches.amin(dim=(2,3), keepdim=True)
    maxv = patches.amax(dim=(2,3), keepdim=True)
    return (patches - minv) / (maxv - minv + 1e-6)
# ============================================================
# PATIENT PROCESSING
# ============================================================
@torch.no_grad()
def process_patient(zpath, dfp, model, out_csv, write_header):

    z = zarr.open(zpath, "r")
    _, H, W = z.shape

    fout = open(out_csv, "a")

    cell_pbar = tqdm(
        total=len(dfp),
        desc="Cells",
        dynamic_ncols=True
    )

    for ty in range(0, H, TILE_SIZE):
        for tx in range(0, W, TILE_SIZE):

            mask = (
                (dfp.cx >= tx) & (dfp.cx < tx + TILE_SIZE) &
                (dfp.cy >= ty) & (dfp.cy < ty + TILE_SIZE)
            )
            sdf = dfp[mask]
            if len(sdf) == 0:
                continue

            tile_np = z[:, ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]
            tile = torch.from_numpy(tile_np.astype(np.float32)).to(DEVICE).half()
            
            centers = torch.stack([
                torch.tensor(sdf.cx.values - tx),
                torch.tensor(sdf.cy.values - ty)
            ], dim=1).to(DEVICE)

            for i in range(0, len(sdf), GPU_BATCH_SIZE):
                sub_centers = centers[i:i+GPU_BATCH_SIZE]
                idxs = sdf.index[i:i+GPU_BATCH_SIZE]

                patches = extract_patches_fast(tile, sub_centers)
                patches = torch.arcsinh(patches / ARCSINH_COFACTOR)
         
                emb = model(patches, return_embedding=True)
                emb = emb.float().cpu().numpy()  # (B, 512)

                for j, idx in enumerate(idxs):
                    row = dfp.loc[idx]
                    meta = {
                        "cell_index": idx,
                        "Patient": row.Patient,
                        "Tissue": row.Tissue,
                        "Class0": row.Class0,
                        "cx": row.cx,
                        "cy": row.cy,
                    }
                    for d in range(OUT_DIM):
                        meta[f"emb_{d}"] = float(emb[j, d])

                    if write_header[0]:
                        fout.write(",".join(meta.keys()) + "\n")
                        write_header[0] = False

                    fout.write(",".join(map(str, meta.values())) + "\n")

                cell_pbar.update(len(idxs))
                cell_pbar.set_postfix(
                    speed=f"{cell_pbar.format_dict['rate']:.1f} cells/s"
                )

            del tile
            torch.cuda.empty_cache()

    cell_pbar.close()
    fout.close()
    gc.collect()


@torch.no_grad()
def robust_minmax_tile(tile, p_low=1.0, p_high=99.0):
    """
    tile: (C, H, W) float16 tensor on GPU
    returns: normalized tile in [0,1]
    """
    C = tile.shape[0]
    tile_flat = tile.view(C, -1)

    lo = torch.quantile(tile_flat, p_low / 100.0, dim=1, keepdim=True)
    hi = torch.quantile(tile_flat, p_high / 100.0, dim=1, keepdim=True)

    tile_flat = torch.clamp(tile_flat, lo, hi)
    tile_flat = (tile_flat - lo) / (hi - lo + 1e-6)

    return tile_flat.view_as(tile)


class MarkerAwareAdapter(nn.Module):
    def __init__(self, in_ch=38):
        super().__init__()

        # Marker mixing
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 1),
            nn.GELU(),
        )

        # Channel attention (SE-style)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 32, 1),
            nn.Sigmoid(),
        )

        # Project to RGB
        self.to_rgb = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x = self.mix(x)
        x = x * self.attn(x)   # marker weighting
        return self.to_rgb(x)
    

# ============================================================
# MAIN
# ============================================================
def main():
    global GPU_BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells_csv", default="/mnt/volumec/Aswath/selected_samples.csv")
    parser.add_argument("--root_dir",  default="/mnt/volumec/Aswath/processed_data/data")
    parser.add_argument(
        "--out_csv",
        default="/mnt/volumec/Aswath/patchmodel/s3cima/CNNResnetembeddings/128markerawarebatchsize1024.csv"
    )
    parser.add_argument("--batchsize",  type=int,  default="128")
    args = parser.parse_args()
    
    GPU_BATCH_SIZE = args.batchsize
    

    df = pd.read_csv(args.cells_csv)
    df["cx"] = (df.XMin + df.XMax) / 2
    df["cy"] = (df.YMin + df.YMax) / 2

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    OUT_DIM = 512    
    NUM_CLASSES = 3
    EMB_DIM = 512
    # EMB_DIM = 2048 # resnet 50  

    model = MultiHeadResNet18()
    model = model.to(DEVICE).half().eval()
    write_header = [True]

    for pid in sorted(df.Patient.unique()):
        print(f"\n=== Processing Patient {pid} ===")
        dfp = df[df.Patient == pid]
        zpath = os.path.join(args.root_dir, str(pid), "data.zarr")
        process_patient(zpath, dfp, model, args.out_csv, write_header)

    print("\n✅ DONE — Fast offline CNN embeddings extracted.")

if __name__ == "__main__":
    main()