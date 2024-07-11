from util.loader import load_data_traditional_inpainting, load_traditional_inpainting_method

import logging
import coloredlogs
import argparse

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--traditional_method', default='DullRazor', type=str)
    # parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--resize', default=True, type=bool)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--color_space', default='rgb', type=str)
    parser.add_argument('--num_channel', default=1, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='hair inpainting with traditional methods in skin lesions images')
args = parse_arguments(parser)

imgs, masks, input_size, img_names, mask_names=load_data_traditional_inpainting(resize=args.resize, img_row=args.img_size, img_col=args.img_size, color_spc=args.color_space)

load_traditional_inpainting_method (traditional_method_name=args.traditional_method, imgs=imgs, masks=masks, imgs_names=img_names)