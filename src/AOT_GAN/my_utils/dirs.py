from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
DIR_PROJECT_EXPERIMENTS = Path.joinpath(PROJECT_DIR, 'models', 'inpainting')
DIR_PROJECT_EXPERIMENTS_MASKS_ATTNET = Path.joinpath(PROJECT_DIR, 'models', 'inpainting', 'masks_AttNet', 'AOT_GAN')


# DIR_PROJECT_DATA_RAW_CA_ISIC18 = Path.joinpath(PROJECT_DIR, 'data', 'CA-ISIC18','raw')
# DIR_PROJECT_DATA_MASKS_CA_ISIC18 = Path.joinpath(PROJECT_DIR, 'data', 'CA-ISIC18', 'mask')

DIR_PROJECT_DATA_RAW = Path.joinpath(PROJECT_DIR, 'data', 'inpainting','raw_without_hair')
DIR_PROJECT_DATA_MASKS = Path.joinpath(PROJECT_DIR, 'data', 'inpainting', 'masks')
DIR_PROJECT_DATA_RAW_HAIR = Path.joinpath(PROJECT_DIR, 'data', 'inpainting', 'raw_with_hair')
DIR_PROJECT_DATA_REPORTS_AOTGAN=Path.joinpath(PROJECT_DIR, 'reports', 'inpainting','aot_gan', 'models')

# DIR_PROJECT_GRAPHS_SEGMENTATION = Path.joinpath(PROJECT_DIR, 'reports', 'hair-segmentation', 'Loss-graphs')
# DIR_PROJECT_GRAPHS_INPAINTING = Path.joinpath(PROJECT_DIR, 'reports', 'inpainting', 'Loss-graphs')

# DIR_PROJECT_HIST_HAIR_SEGMENTATION = Path.joinpath(PROJECT_DIR, 'reports', 'hair-segmentation', 'historials')
# DIR_PROJECT_HIST_INPAINTING = Path.joinpath(PROJECT_DIR, 'reports', 'inpainting', 'historials')


# DIR_PROJECT_DATA_SEGMENTED_COMBINATION = Path.joinpath(PROJECT_DIR, 'reports', 'hair-segmentation','overlay')
# DIR_PROJECT_DATA_SEGMENTED_MASKS = Path.joinpath(PROJECT_DIR, 'reports', 'hair-segmentation','masks')

# DIR_PROJECT_DATA_INPAINTED_RESULTS = Path.joinpath(PROJECT_DIR, 'reports', 'inpainting','results')

DIR_PROJECT_DATA_INPAINTED_RESULTS = Path.joinpath(PROJECT_DIR, 'reports', 'inpainting','aot_gan')
