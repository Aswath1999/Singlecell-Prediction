import os, json,time
import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import visualization.histogram as hist 
import utils.helper as utils


# ----------------------------- Arcsinh utils ------------------------------

def asinh_transform(x: np.ndarray, cofactor: float = 5.0, eps: float = 0.0) -> np.ndarray:
    """
    Numerically stable arcsinh transform used in cytometry / imaging.
    y = asinh((x + eps) / cofactor)

    cofactor: larger => gentler compression (try 5, 10, 20)
    eps: small offset that can keep tiny positives from being crushed
    """
    x = np.asarray(x, dtype=np.float32)
    c = float(cofactor) if cofactor > 0 else 5.0
    return np.arcsinh((x + eps) / c)


def choose_cofactor(x: np.ndarray, method: str = "p99", min_c: float = 1.0) -> float:
    """
    Choose a per-channel cofactor from raw intensities.
    method: "p95", "p99" (default), or "median_pos"
    Uses only positive values to avoid background bias.
    """
    x = np.asarray(x, dtype=np.float32)
    pos = x[x > 0]
    if pos.size == 0:
        return max(min_c, 1.0)
    if method == "median_pos":
        c = float(np.median(pos))
    elif method == "p95":
        c = float(np.percentile(pos, 95))
    else:  # "p99"
        c = float(np.percentile(pos, 99))
    return max(c, min_c)



# ----------------------------- Per-sample QC ------------------------------

def sample_hist(root_path, sample_name, qc_path, bins=100, sample_max=10000,
                use_asinh=False, cofactor_mode="p99", fixed_cofactor=None):
    """
    Draw histograms for a single sample; optionally arcsinh-transform before plotting.
    """
    marker_stats_dict = get_channel_data(
        root_path, sample_name, sample_max=sample_max, tile_size=None,
        use_asinh=use_asinh, cofactor_mode=cofactor_mode, fixed_cofactor=fixed_cofactor
    )
    qc_save_path = f'{qc_path}/{sample_name}'
    os.makedirs(qc_save_path, exist_ok=True)
    hist.multipanel_hist(marker_stats_dict, save_path=f'{qc_save_path}/channel_hist',
                         bins=bins, title=sample_name, xlabel='Marker Intensity', ylabel='Frequency')
    hist.multipanel_hist(marker_stats_dict, save_path=f'{qc_save_path}/channel_hist_log',
                         bins=bins, title=sample_name, xlabel='Marker Intensity',
                         ylabel='log(Frequency)', log_y=True)





# ---------------------------- Global QC / Stats ---------------------------

def global_hist(root_path, sample_list, qc_save_path, bins=100, sample_max=100000,
                use_asinh=False, cofactor_mode="p99", fixed_cofactor=None):
    """
    Build + plot global histograms across many samples (optionally arcsinh first).
    """
    print('Generating global histogram')
    qc_save_path = f'{qc_save_path}/global'
    os.makedirs(qc_save_path, exist_ok=True)
    merged_marker_stats_dict = gen_global_dist_data(
        root_path, sample_list, qc_save_path, sample_max=sample_max, tile_size=None,
        use_asinh=use_asinh, cofactor_mode=cofactor_mode, fixed_cofactor=fixed_cofactor
    )
    plot_hist(merged_marker_stats_dict, save_path=f'{qc_save_path}/channel_hist')


def plot_hist(merged_marker_stats_dict, save_path, bins=100, title='Global',
              xlabel='Marker Intensity', ylabel='Frequency'):
    hist.multipanel_hist(merged_marker_stats_dict, save_path=save_path,
                         bins=bins, title=title, xlabel=xlabel, ylabel=ylabel)
    hist.multipanel_hist(merged_marker_stats_dict, save_path=save_path + '_log',
                         bins=bins, title=title + ' log frequency',
                         xlabel=xlabel, ylabel='log(Frequency)', log_y=True)



def calculate_normalization_stats(root_path, sample_list, qc_save_path,
                                  sample_max=100000, tile_size=None,
                                  use_asinh=True, cofactor_mode="p99", fixed_cofactor=None):
    """
    Compute global distributions (optionally arcsinh), then z-score normalize.
    Saves:
      - global_hist*.npz
      - normalization_stats.csv (mean, std per marker after chosen transform)
      - pre/post normalization plots
    """
    qc_save_path = f'{qc_save_path}/normalization'
    os.makedirs(qc_save_path, exist_ok=True)

    data_dict = gen_global_dist_data(
        root_path, sample_list, qc_save_path, sample_max=sample_max, tile_size=tile_size,
        use_asinh=use_asinh, cofactor_mode=cofactor_mode, fixed_cofactor=fixed_cofactor
    )

    tag = f'_tiled_{tile_size}' if tile_size is not None else ''
    plot_hist(data_dict, save_path=f'{qc_save_path}/channel_hist{tag}')

    data_dict, norm_stats_df = normalize_channels(data_dict)
    norm_stats_df.to_csv(f'{qc_save_path}/normalization_stats{tag}.csv', index=False)

    plot_hist(data_dict, save_path=f'{qc_save_path}/channel_hist{tag}_normalized')


def normalize_channels(data_dict):
    """
    Z-score normalize each marker distribution in-place (mean 0, std 1).
    Returns the normalized dict and a small DataFrame with per-marker mean/std.
    """
    norm_stats_list = []
    for key, data in data_dict.items():
        data = np.asarray(data, dtype=np.float32)
        mean = float(np.mean(data)) if data.size else 0.0
        std  = float(np.std(data))  if data.size else 1.0
        std  = std if std > 0 else 1.0
        data_dict[key] = (data - mean) / std
        norm_stats_list.append({'marker': key, 'mean': mean, 'std': std})
    return data_dict, pd.DataFrame(norm_stats_list)




# --------------------------- Data aggregation -----------------------------

def gen_global_dist_data(root_path, sample_list, qc_save_path,
                         sample_max=100000, tile_size=None,
                         save_every_markers=8,  # write partial npz every N markers
                         progress_fname="progress.json",
                         use_asinh=False, cofactor_mode="p99", fixed_cofactor=None):
    """
    Build global per-marker distributions across samples with visible progress and
    periodic partial saves so you can monitor long runs.
    """
    if tile_size is None:
        save_file_path = f'{qc_save_path}/global_hist.npz'
    else:
        save_file_path = f'{qc_save_path}/global_hist_tiled_{tile_size}.npz'
    os.makedirs(qc_save_path, exist_ok=True)

    # If final output already exists, load & return
    if os.path.exists(save_file_path):
        print('Loading global histogram data as a dictionary')
        data = np.load(save_file_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    merged = {}
    total_samples = len(sample_list)
    t0 = time.time()

    # simple progress json
    progress_path = os.path.join(qc_save_path, progress_fname)
    def write_progress(**kw):
        kw.setdefault("time_sec", round(time.time() - t0, 1))
        with open(progress_path, "w") as f:
            json.dump(kw, f, indent=2)

    write_progress(status="starting",
                   total_samples=total_samples,
                   done_samples=0,
                   markers_in_merged=0)

    # pass 1: seed marker names from the first pass through samples
    for s_idx, sample_name in enumerate(tqdm(sample_list, desc="Collecting markers (first pass)")):
        marker_stats = get_channel_data(
            root_path, sample_name, sample_max=max(1, sample_max // max(1, total_samples)),
            tile_size=tile_size, use_asinh=use_asinh,
            cofactor_mode=cofactor_mode, fixed_cofactor=fixed_cofactor
        )
        for m in marker_stats.keys():
            merged.setdefault(m, [])
        write_progress(status="seeding_markers",
                       done_samples=s_idx + 1,
                       markers_in_merged=len(merged))

    # pass 2: aggregate values
    marker_names = list(merged.keys())
    for s_idx, sample_name in enumerate(tqdm(sample_list, desc="Aggregating marker values (per sample)")):
        marker_stats = get_channel_data(
            root_path, sample_name, sample_max=max(1, sample_max // max(1, total_samples)),
            tile_size=tile_size, use_asinh=use_asinh,
            cofactor_mode=cofactor_mode, fixed_cofactor=fixed_cofactor
        )

        for i, m in enumerate(tqdm(marker_names, desc=f"  {sample_name}", leave=False)):
            if m in marker_stats:
                merged[m].extend(marker_stats[m])

            if (i + 1) % save_every_markers == 0:
                tmp_path = save_file_path + ".partial"
                np.savez(tmp_path, **{k: np.array(v, dtype=np.float32) for k, v in merged.items()})
                write_progress(status="partial_save",
                               sample=sample_name,
                               done_samples=s_idx + 1,
                               markers_saved=i + 1,
                               markers_in_merged=len(merged),
                               partial_npz=tmp_path)

        write_progress(status="sample_done",
                       sample=sample_name,
                       done_samples=s_idx + 1,
                       markers_in_merged=len(merged))

    # final save
    np.savez(save_file_path, **{k: np.array(v, dtype=np.float32) for k, v in merged.items()})
    write_progress(status="done",
                   final_npz=save_file_path,
                   markers_in_merged=len(merged))

    # quick overview figure
    plot_hist(merged, save_path=f'{qc_save_path}/channel_hist')

    return merged



# --------------------------- Channel extraction ----------------------------

def get_channel_data(root_path, sample_name, sample_max, tile_size,
                     use_asinh: bool = True,
                     cofactor_mode: str = "p99",
                     fixed_cofactor: float | None = None,
                     eps: float = 0.0):
    """
    Return per-marker vectors from the zarr (optionally restricted to tiles),
    with optional arcsinh compression applied channel-wise.

    Returns dict: {marker_name -> 1D np.array}
    """
    data, channels = utils.load_zarr_w_channel(root_path, sample_name)
    marker_channel_dict = dict(zip(channels['marker'], channels['channel']))

    marker_stats_dict = {}
    rng = np.random.default_rng(12345)  # deterministic subsampling

    for marker, channel in marker_channel_dict.items():
        # 1) Pull raw values (full image or only tiles)
        if tile_size is None:
            vec = np.asarray(data[channel, :, :], dtype=np.float32).ravel()
        else:
            vec = get_tile_region(data[channel, :, :], root_path, sample_name, tile_size).astype(np.float32)

        # 2) Optional subsample for speed
        if vec.size > sample_max:
            idx = rng.choice(vec.size, size=sample_max, replace=False)
            vec = vec[idx]

        # 3) Optional arcsinh
        if use_asinh:
            if fixed_cofactor is not None:
                c = float(fixed_cofactor)
            else:
                c = choose_cofactor(vec, method=cofactor_mode, min_c=1.0)
            vec = asinh_transform(vec, cofactor=c, eps=eps)

        # 4) Store
        marker_stats_dict[marker] = vec

    return marker_stats_dict



# ---------------------------- Tile region helper ---------------------------

def get_tile_region(channel_img, root_path, sample_name, tile_size):
    """
    Concatenate all tile pixel values (h,w) windows from this channel into one vector.
    """
    tile_df = utils.load_tile_info(root_path, sample_name, tile_size)
    tile_regions = []
    for _, row in tile_df.iterrows():
        x, y = int(row['h']), int(row['w'])
        tile_regions.append(channel_img[y:y+tile_size, x:x+tile_size].ravel())
    if len(tile_regions) > 0:
        return np.concatenate(tile_regions, axis=0)
    else:
        return np.empty((0,), dtype=channel_img.dtype)
    
   