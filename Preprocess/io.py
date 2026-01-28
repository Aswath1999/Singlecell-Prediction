import os
import numpy as np
import tifffile
import zarr
import logging

# Configure logging to save to a file
logging.basicConfig(filename='/mnt/volumec/Aswath/CANVAS/preprocessing.log', level=logging.INFO)

def _marker_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base.split('_')[-1]  # e.g., "..._CD38" -> "CD38"

def _read_plane(path: str) -> np.ndarray:
    try:
        img = tifffile.imread(path)
    except Exception:
        # robust fallback
        with tifffile.TiffFile(path) as tf:
            try:    img = tf.asarray(key=0)
            except: img = tf.pages[0].asarray()
    img = np.squeeze(np.asarray(img))           #gives only grayscale H*W
    print("img shape",img.shape , flush=True)
    logging.info(f"Read image from {path} with shape {img.shape}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D TIFF, got {img.shape} for {path}")
    return img


def stack_sample_to_zarr(input_sample_dir: str,
                         sample_name: str,
                         output_sample_dir: str,
                         dummy_sample_dir: str,
                         input_ext: str = 'tif',
                         chunk_hw: int = 256,
                         create_dummies: bool = True):
    """
    Build ONE Zarr cube (C,H,W) for this sample from MANY single-channel TIFFs,
    using EXACT name matching between TIFF marker tokens and <sample>.txt.
    """
    # 1) channel order (exact names)
    ch_txt = os.path.join(input_sample_dir, f"{sample_name}.txt")
    if not os.path.exists(ch_txt):
        raise FileNotFoundError(f"Missing channel list: {ch_txt}")
    with open(ch_txt) as f:
        channels = [ln.strip() for ln in f if ln.strip()]
    C = len(channels)

    #2  collect tiffs
    tifs = [os.path.join(input_sample_dir,f) for f in os.listdir(input_sample_dir) if f.endswith('.'+input_ext.lower()) ]
    if(len(tifs) == 0):
        raise FileNotFoundError(f"No .{input_ext} files in {input_sample_dir}")
    
    # 3) map marker->path (EXACT)
    marker_to_path = {} 
    for p in tifs:
        m=_marker_from_path(p)
        if m in marker_to_path:
            # same marker appears more than once → warn early
            raise ValueError(f"Duplicate TIFFs for marker '{m}' in {input_sample_dir}")
        marker_to_path[m] = p
    print("marker to path",marker_to_path)
    
    # 4) reference size from any TIFF (ideally DNA channel if present)
    
    ref_path = marker_to_path.get('HOECHST 2', None) or next(iter(marker_to_path.values()))
    ref_img = _read_plane(ref_path)
    H,W = ref_img.shape
    
    # 5) create root array once (Zarr root is the array, not a group key)
    os.makedirs(output_sample_dir, exist_ok=True)
    zarr_path = os.path.join(output_sample_dir, 'data.zarr') 
    store = zarr.DirectoryStore(zarr_path)
    if os.path.exists(zarr_path):
        z = zarr.open_array(store, mode='a')
        if z.shape != (C, H, W):
            z = zarr.create(
            shape=(C, H, W),
            chunks=(1, min(chunk_hw, H), min(chunk_hw, W)),
            dtype=ref_img.dtype if np.issubdtype(ref_img.dtype, np.integer) else np.int32,
            store=store,
            overwrite=True
        )
            
        logging.info(f"Zarr already exists at {zarr_path}, overwriting data")
           
    else:
        z = zarr.create(
            shape=(C, H, W),
            chunks=(1, min(chunk_hw, H), min(chunk_hw, W)),
            dtype=ref_img.dtype if np.issubdtype(ref_img.dtype, np.integer) else np.int32,
            store=store,
            overwrite=True
        )
    # 6) write planes in the exact channel order
    missing = []
    for idx, ch_name in enumerate(channels):
        if ch_name in marker_to_path:
            img = _read_plane(marker_to_path[ch_name])
            if img.shape != (H, W):
                raise ValueError(f"Size mismatch for {marker_to_path[ch_name]}: {img.shape} vs {(H,W)}")
            z[idx, :, :] = img
        else:
            missing.append(ch_name)
            
    if missing:
       
        # Fail fast with actionable information
        have = sorted(marker_to_path.keys())
        need = missing
        raise ValueError(
            f"[{sample_name}] Missing TIFFs for channels (exact name match required): {need}\n"
            f"Present markers from filenames: {have}\n"
            f"Check <sample>.txt entries and TIFF basenames' last tokens."
        )
        
    # 7) channels.csv (once per sample)
    ch_csv = os.path.join(output_sample_dir, 'channels.csv')
    if not os.path.exists(ch_csv):
        with open(ch_csv, 'w') as f:
            f.write('channel,marker\n')
            for i, m in enumerate(channels):
                f.write(f'{i},{m}\n')

    # 8) dummies (optional, per TIFF)
    if create_dummies:
        os.makedirs(dummy_sample_dir, exist_ok=True)
        for p in tifs:
            base = os.path.basename(p)
            dummy = os.path.join(dummy_sample_dir, base + f".dummy_{input_ext}")
            if not os.path.exists(dummy):
                with open(dummy, 'w') as f:
                    f.write('')


    
    

def tiff_to_zarr(input_path, output_path, dummy_input_path, file_name, input_ext='tif', chunk_size=(None,256,256)):
    # resolve input
    input_file_rel = file_name if file_name.lower().endswith('.'+input_ext) else f"{file_name}.{input_ext}"
    input_file = os.path.join(input_path, input_file_rel)

    # sample folder/name and channel list
    rel_dir     = os.path.dirname(file_name)
    sample_name = os.path.basename(rel_dir) or os.path.splitext(os.path.basename(file_name))[0]
    ch_file = os.path.join(input_path, rel_dir, f"{sample_name}.txt")
    with open(ch_file) as f:
        channels = [ln.strip() for ln in f if ln.strip()]
    C = len(channels)

    # read this channel image
    img = _read_single_plane_tif(input_file)
    H, W = img.shape
    cidx = _marker_index_from_filename(channels, input_file_rel)

    # zarr store (ROOT ARRAY, not a group)
    sample_out_dir = os.path.join(output_path, rel_dir)
    os.makedirs(sample_out_dir, exist_ok=True)
    out_zarr = os.path.join(sample_out_dir, "data.zarr")
    store = zarr.DirectoryStore(out_zarr)

    if os.path.exists(out_zarr):
        z = zarr.open_array(store, mode='a')          # root array
        if z.shape != (C, H, W):
            raise ValueError(f"{out_zarr} shape {z.shape} != expected {(C,H,W)}")
    else:
        dtype = img.dtype if np.issubdtype(img.dtype, np.integer) else np.int32
        z = zarr.create(
            shape=(C, H, W),
            chunks=(1, min(chunk_size[1],H), min(chunk_size[2],W)),
            dtype=dtype,
            store=store,
            overwrite=True,
        )

    # write this slice
    z[cidx, :, :] = img

    # channels.csv (once)
    ch_csv = os.path.join(sample_out_dir, 'channels.csv')
    if not os.path.exists(ch_csv):
        with open(ch_csv, 'w') as f:
            f.write('channel,marker\n')
            for i, m in enumerate(channels):
                f.write(f'{i},{m}\n')

    # dummy file (mirrors relative path)
    dummy_dir = os.path.join(dummy_input_path, rel_dir)
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_path = os.path.join(dummy_dir, os.path.basename(input_file_rel) + f".dummy_{input_ext}")
    if not os.path.exists(dummy_path):
        with open(dummy_path, 'w') as f:
            f.write('')


def mcd_to_zarr(input_path, output_path, dummy_input_path, file_name, input_ext='mcd', chunk_size=(None, 256, 256)):
    """
    Optional: for IMC .mcd inputs (kept here for completeness).
    Creates per-acquisition zarrs and dummies; expects <file_name>.txt alongside the .mcd.
    """
    import pyimc

    # resolve input .mcd
    if file_name.lower().endswith(f'.{input_ext}'.lower()):
        input_file_rel = file_name
        file_stem_rel = os.path.splitext(file_name)[0]
    else:
        input_file_rel = f"{file_name}.{input_ext}"
        file_stem_rel = file_name

    input_file = os.path.join(input_path, input_file_rel)

    # channel list (same folder as .mcd)
    input_channel_file = os.path.join(input_path, os.path.dirname(file_name), f"{os.path.basename(file_stem_rel)}.txt")
    if not os.path.exists(input_channel_file):
        raise FileNotFoundError(f"Channel file does not exist at {input_channel_file}")
    with open(input_channel_file, 'r') as f:
        channels = [ln.strip() for ln in f if ln.strip()]

    data = pyimc.Mcd.parse(input_file)
    acquisition_ids = data.acquisition_ids()

    for acquisition_id in acquisition_ids:
        # extract (C,H,W) for this acquisition
        img_data, label_list = extract_acquisition(data, acquisition_id)  # expects (C,H,W)

        # sanity check C
        if img_data.ndim != 3:
            raise ValueError(f"Expected (C,H,W) from .mcd acquisition; got {img_data.shape}")
        C, H, W = img_data.shape
        if C != len(channels):
            raise ValueError(f"Acquisition has {C} channels but {len(channels)} provided in channel list: {input_channel_file}")

        # per-acquisition outputs (mirror nested)
        rel_dir = os.path.dirname(file_name)
        out_dir = os.path.join(output_path, rel_dir + f"_acquisition_{acquisition_id}")
        os.makedirs(out_dir, exist_ok=True)

        out_zarr = os.path.join(out_dir, "data.zarr")
        if os.path.exists(out_zarr):
            print(f'Zarr file already exists at {out_zarr}')
        else:
            zarr.array(img_data, chunks=(1, min(chunk_size[1], H), min(chunk_size[2], W)), store=out_zarr)

        # channels.csv
        channels_csv = os.path.join(out_dir, 'channels.csv')
        if not os.path.exists(channels_csv):
            with open(channels_csv, 'w') as f:
                f.write('channel,marker\n')
                for i, m in enumerate(channels):
                    f.write(f'{i},{m}\n')

        # dummy
        dummy_dir = os.path.join(dummy_input_path, rel_dir)
        os.makedirs(dummy_dir, exist_ok=True)
        dummy_file = os.path.join(dummy_dir, f"{os.path.basename(file_stem_rel)}_acquisition_{acquisition_id}.dummy_{input_ext}")
        if not os.path.exists(dummy_file):
            with open(dummy_file, 'w') as f:
                f.write('')




