# IMPORTS
from util.loader import load_raw_images, generate_predictions

import logging
import coloredlogs
import argparse

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--model', default='U-Net', type=str)
    parser.add_argument('--resize', default=True, type=bool)
    parser.add_argument('--img_size', default=128, type=int)
    # parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--color_space', default='ycrcb', type=str)
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='hair segmentation in skin lesions images')
args = parse_arguments(parser)

imgs_orig, img_shape, img_names=load_raw_images(resize=args.resize, img_row=args.img_size, img_col=args.img_size, color_spc=args.color_space, allowed_extensions=('png', 'jpg', 'jpeg'))

generate_predictions(model_name=args.model, images=imgs_orig, img_names=img_names, idx=args.seed, color_space=args.color_space)

