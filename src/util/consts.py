from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]

PATH_PROJECT_DATA_HR_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'segmentation_hair','raw')
PATH_PROJECT_DATA_HR_MASKS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'segmentation_hair', 'masks')

PATH_PROJECT_SEGMENTATION_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'segmentation', 'models')
PATH_PROJECT_SEGMENTATION_HIST = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'segmentation', 'history')
PATH_PROJECT_SEGMENTATION_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'segmentation', 'metrics')
PATH_PROJECT_SEGMENTATION_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'segmentation', 'masks')

DIR_PROJECT_DATA_MASKS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'inpainting', 'masks')
DIR_PROJECT_DATA_RAW_HAIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'inpainting', 'raw_with_hair')

PATH_PROJECT_TRADITIONAL_INPAINTING_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'inpainting', 'traditional_methods')

PATH_PROJECT_DATA_ISIC = Path.joinpath(PATH_PROJECT_DIR, 'data', 'classification','ISIC-2020')
PATH_PROJECT_DATA_DERM7PT = Path.joinpath(PATH_PROJECT_DIR, 'data', 'classification','Derm7pt')
PATH_PROJECT_DATA_PH2 = Path.joinpath(PATH_PROJECT_DIR, 'data', 'classification','PH2')

PATH_PROJECT_CLASSIFICATION_MODELS=Path.joinpath(PATH_PROJECT_DIR, 'reports', 'classification', 'models')
PATH_PROJECT_CLASSIFICATION_HIST = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'classification', 'history')
PATH_PROJECT_CLASSIFICATION_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'classification', 'metrics')
