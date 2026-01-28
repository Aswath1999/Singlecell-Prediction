import os, glob
from tqdm import tqdm
import utils.helper as utils
import pdb
from utils import helper
from types import SimpleNamespace
import visualization.multiplex_image as multi_img
from preprocess.label_tiles_from_tissues import label_tiles_from_tissues

def run_preprocess(config):
    output_path = os.path.join(config.data_root, 'processed_data')
    data_path = f'{output_path}/data'
   
    qc_path = f'{output_path}/qc'
    image_path = os.path.join(config.data_root, config.input_path, config.raw_image_path)
    print("found image path: ", image_path)
    dummy_input_path = os.path.join(config.data_root, config.input_path, 'dummy_input')
    os.makedirs(dummy_input_path, exist_ok=True)

    # Step 1: Convert tiff to zarr and QC
    zarr_conversion(config.data_root, image_path, config.input_ext, data_path, qc_path, dummy_input_path)
    


    # Step 2: Choose channels with colormap and visualize individual images
    visualize_samples(
    config.data_type, data_path, output_path,
    config.config_root, config.selected_channel_color_file, config.channel_strength_file
)
    
    print("vis done")

    # Step 3: Generate tiles
    ROI_path = config.ROI_path if hasattr(config, 'ROI_path') else None
    selected_region = config.selected_region if hasattr(config, 'selected_region') else None
    tiling(dummy_input_path, config.input_ext, data_path, ROI_path, 
           config.inference_window_um, 
           config.input_pixel_per_um, 
           selected_region, 
           config.ref_channel)
    
    print('tiling done')

    # Step 4: QC
    normalization(dummy_input_path, config.input_ext, data_path, config.inference_window_um, config.input_pixel_per_um, qc_path)
    
    pritn("normalization done")

    # Step 5: Copy common_channels.txt to processed_data/data
    import shutil
    common_channels_path = os.path.join(config.data_root, 'MAIT STUDY', 'common_channels.txt')
    shutil.copy(common_channels_path, os.path.join(data_path, 'common_channels.txt'))


def normalization(input_path, input_ext, data_path, inference_window_um, input_pixel_per_um, qc_path):
    from preprocess import qc

    # We only need sample names that have a data.zarr ready
    sample_names = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and
           os.path.exists(os.path.join(data_path, d, 'data.zarr'))
    ])
    print(f"[normalization] samples with zarr: {sample_names}")

    qc_save_path = os.path.join(qc_path, 'global')
    os.makedirs(qc_save_path, exist_ok=True)

    training_window_um = inference_window_um * 2
    training_window_pixel = int(training_window_um * input_pixel_per_um)

    # qc.calculate_normalization_stats(
    #     data_path, sample_names, qc_save_path, tile_size=training_window_pixel
    # )
    qc.calculate_normalization_stats(
    data_path, sample_names, qc_save_path,
    tile_size=training_window_pixel,
    use_asinh=True,          # <- enable arcsinh
    cofactor_mode="p99",     # or "p95" / "median_pos"
    fixed_cofactor=None      # or set a number like 5.0 to force same cofactor for all
)

def tiling(input_path, input_ext, data_path, ROI_path,
           inference_window_um, input_pixel_per_um,
           selected_region, ref_channel):

    from preprocess.tile import gen_tiles
    import os
    from tqdm import tqdm

    # compute pixel window sizes
    training_window_um   = inference_window_um * 2
    training_window_px   = int(round(training_window_um * input_pixel_per_um))
    inference_window_px  = int(round(inference_window_um * input_pixel_per_um))

    # discover all samples
    sample_names = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and
           os.path.exists(os.path.join(data_path, d, 'data.zarr'))
    ])
    print(f"[tiling] Found {len(sample_names)} samples: {sample_names}")

    # path to your per-cell annotation CSV
    # cell_csv = "/mnt/volumec/Aswath/selected_samples.csv"
    cell_csv = "/mnt/volumec/Aswath/HCC_noCD8junk.csv"

    for slideID in tqdm(sample_names, desc="Tiling samples"):
        tiles_dir = os.path.join(data_path, slideID, "tiles")

        pos_train = os.path.join(tiles_dir, f"positions_{training_window_px}.csv")
        pos_infer = os.path.join(tiles_dir, f"positions_{inference_window_px}.csv")

        # Training tiles
        if not os.path.exists(pos_train):
            gen_tiles(
                data_path=data_path,
                slideID=slideID,
                ref_channel=ref_channel,
                ROI_path=ROI_path,
                tile_size=training_window_px,
                selected_region=selected_region
            )

        # Label the training tiles
        label_tiles_from_tissues(
            data_path=data_path,
            slideID=slideID,
            cell_csv_path=cell_csv,
            tile_size=training_window_px,
            purity_thresh=0.7
        )

        # Inference tiles
        if not os.path.exists(pos_infer):
            gen_tiles(
                data_path=data_path,
                slideID=slideID,
                ref_channel=ref_channel,
                ROI_path=ROI_path,
                tile_size=inference_window_px,
                selected_region=selected_region
            )

        # Label the inference tiles
        label_tiles_from_tissues(
            data_path=data_path,
            slideID=slideID,
            cell_csv_path=cell_csv,
            tile_size=inference_window_px,
            purity_thresh=0.7
        )
# def visualize_samples(data_type, input_path, input_ext, output_path, config_root, selected_channel_color_file, channel_strength_file, data_path):
#     selected_channel_color_file = os.path.join(config_root, selected_channel_color_file)
#     channel_strength_file = os.path.join(config_root, channel_strength_file)
#     utils.visualize_color_yaml_file(selected_channel_color_file, output_path) # Confirm color
#     color_dict = utils.load_channel_yaml_file(selected_channel_color_file)
#     strength_dict = utils.load_channel_yaml_file(channel_strength_file)
#     file_names = utils.get_file_name_list(input_path, input_ext)
#     for file_name in tqdm(file_names):
#         multi_img.visualize_sample(data_path, file_name, color_dict, strength_dict, data_type)
        

def visualize_samples(data_type, data_path, output_path, config_root,
                      selected_channel_color_file, channel_strength_file):
    selected_channel_color_file = os.path.join(config_root, selected_channel_color_file)
    channel_strength_file = os.path.join(config_root, channel_strength_file)
    utils.visualize_color_yaml_file(selected_channel_color_file, output_path)

    color_dict    = utils.load_channel_yaml_file(selected_channel_color_file)
    strength_dict = utils.load_channel_yaml_file(channel_strength_file)

    # discover samples by presence of data.zarr
    sample_names = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and
           os.path.exists(os.path.join(data_path, d, 'data.zarr'))
    ])

    for sample_name in tqdm(sample_names):
        multi_img.visualize_sample(data_path, sample_name, color_dict, strength_dict, data_type)
        
       
        
        
def zarr_conversion(data_root, image_path, input_ext, data_path, qc_path, dummy_input_path):
    from preprocess import io   # where stack_sample_to_zarr lives
    from preprocess import qc

    # 1) enumerate sample folders under image_path
    sample_dirs = sorted(
        d for d in os.listdir(image_path)
        if os.path.isdir(os.path.join(image_path, d))
    )
    print(f"[zarr] Found {len(sample_dirs)} sample folders")

    # 2) build each sample zarr in one shot
    for sample_name in tqdm(sample_dirs, desc="Stacking per-sample"):
        input_sample_dir  = os.path.join(image_path, sample_name)
        output_sample_dir = os.path.join(data_path,  sample_name)
        dummy_sample_dir  = os.path.join(dummy_input_path, sample_name)
        os.makedirs(output_sample_dir, exist_ok=True)
        os.makedirs(dummy_sample_dir,  exist_ok=True)

        io.stack_sample_to_zarr(
            input_sample_dir=input_sample_dir,
            sample_name=sample_name,
            output_sample_dir=output_sample_dir,
            dummy_sample_dir=dummy_sample_dir,
            input_ext=input_ext,
            chunk_hw=512,
            create_dummies=True,     # optional here
        )

    # 3) discover samples with data.zarr (sanity)
    sample_names = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and
           os.path.exists(os.path.join(data_path, d, 'data.zarr'))
    )
    print("[zarr] Data zarr present for samples:", sample_names)

    # 4) global QC
    print('[zarr] Generating global QC histogram')
    # qc.global_hist(data_path, sample_names, qc_path)
    qc.global_hist(data_path, sample_names, qc_path, use_asinh=True, cofactor_mode="p99")
    




# def zarr_conversion(data_root, image_path, input_ext, data_path, qc_path, dummy_input_path):
#     from canvas.preprocess import io 
#     from canvas.preprocess import qc
#     # Start preprocessing
#     file_names = utils.get_file_name_list(image_path, input_ext)
#     print("files names: ", file_names)
#     print('Converting tiff to zarr')
#     for file_name in tqdm(file_names):
#         print(file_name)
#         if input_ext in ['qptiff', 'tiff', 'tif']:
#             io.tiff_to_zarr(image_path, data_path, dummy_input_path, file_name, input_ext = input_ext)
#         elif input_ext == 'mcd':
#             io.mcd_to_zarr(image_path, data_path, dummy_input_path, file_name)
#         else:
#             raise ValueError(f'Unsupported input extension: {input_ext}')

#     # Plot global QC histogram
#     print('Generating global QC histogram')
#     qc_file_names = utils.get_file_name_list(dummy_input_path, f'dummy_{input_ext}')
#     qc.global_hist(data_path, qc_file_names, qc_path)