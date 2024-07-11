import os
import math
import numpy as np
from glob import glob
import sys
import re


from random import shuffle
from PIL import Image, ImageFilter
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AOT_GAN'))

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from my_utils.dirs import *
from my_utils.option import args


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type
        print(self.w)

        # image and mask
        self.image_path = []
        self.mask_path = []
        #self.image_masked_path = []
        masks_names = [f for f in os.listdir(DIR_PROJECT_DATA_RAW) if os.path.isfile(os.path.join(DIR_PROJECT_DATA_RAW, f))]
        # print(masks_names)
        names_train, names_test = train_test_split(masks_names,  test_size=0.2, random_state = args.train_seed)
        # print(names_train)
        for filename in os.listdir(args.dir_mask):
            base_name = os.path.splitext(filename)[0]  
            parts = base_name.split('_')  
            desired_part = '_'.join(parts[:-2]) 
            if f'{desired_part}.jpg' in names_train:
                self.mask_path.append(os.path.join(args.dir_mask, filename))
                
                self.image_path.append(os.path.join(args.dir_image, f'{base_name}.jpg'))
            
        def clean_file_paths(file_paths):
            cleaned_paths = []
            for path in file_paths:
                # Utilizamos expresiones regulares para encontrar y reemplazar el patrón deseado
                cleaned_path = re.sub(r'(_[a-z])?_[a-z]?\.', r'.', path)
                cleaned_paths.append(cleaned_path)
            return cleaned_paths

        self.image_path = clean_file_paths(self.image_path)

        print(len(self.image_path))
        print(len(self.mask_path))

        print(self.image_path)
        print(self.mask_path)
            
            
            
            
        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()])
        self.mask_trans = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])
        
        index = np.random.randint(0, len(self.mask_path))
        mask = Image.open(self.mask_path[index])  
        mask = mask.convert('L')
        
        # augment
        image = self.img_trans(image) * 2. - 1.
        mask = F.to_tensor(self.mask_trans(mask))
        
        return image, mask, filename



if __name__ == '__main__': 

    # from attrdict import AttrDict
    # args = {
    #     'dir_image': DIR_PROJECT_DATA_RAW_HAIR,
    #     'dir_mask': DIR_PROJECT_DATA_MASKS,
    #     'image_size': 512
    # }
    # args = AttrDict(args)

    data = InpaintingData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)