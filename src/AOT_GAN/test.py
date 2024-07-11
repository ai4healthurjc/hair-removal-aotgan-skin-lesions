import os
import argparse
import importlib
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from my_utils.option import args
# from .dirs import *


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def main_worker(args, use_gpu=True): 

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    
    # Load model state
    state_dict = torch.load(args.pre_train, map_location='cuda')
    if 'module' in list(state_dict.keys())[0]:
        # The model was saved with DataParallel, remove 'module' from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  # Remove the 'module.' prefix
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model = model.cuda()
    model.eval()

    # Prepare dataset
    image_paths = []
    mask_paths = []
    
    for filename in os.listdir(args.dir_mask):
        mask_paths.append(os.path.join(args.dir_mask, filename))
        basename = os.path.splitext(filename)[0]
        image_paths.append(os.path.join(args.dir_image_test, f'{basename}.jpg'))

    os.makedirs(args.outputs, exist_ok=True)
    
    # Iteration through datasets
    for ipath, mpath in zip(image_paths, mask_paths): 
        # Check if image and mask files exist before trying to open them
        if not os.path.exists(ipath):
            # print(f"El archivo {ipath} no existe.")
            continue
        if not os.path.exists(mpath):
            # print(f"El archivo {mpath} no existe.")
            continue
        
        # Continue only if files exist
        image = ToTensor()(Image.open(ipath).convert('RGB'))
        image  = F.interpolate(image.unsqueeze(0), size=(512,512), mode='bilinear', align_corners=False).squeeze(0)
        image = (image * 2.0 - 1.0).unsqueeze(0)
        mask = ToTensor()(Image.open(mpath).convert('L'))
        mask = mask.unsqueeze(0)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask.float()) + mask
        
        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img
        image_name = os.path.basename(ipath).split('.')[0]
        postprocess(image_masked[0]).save(os.path.join(args.outputs, f'{image_name}_masked.png'))
        postprocess(pred_img[0]).save(os.path.join(args.outputs, f'{image_name}_pred.png'))
        postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f'{image_name}_comp.png'))
        print(f'Se ha guardado en {os.path.join(args.outputs, image_name)}')


if __name__ == '__main__':
    main_worker(args)