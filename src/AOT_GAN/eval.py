import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from multiprocessing import Pool
import os
from my_utils.dirs import *

from metric import metric as module_metric

parser = argparse.ArgumentParser(description='Image Inpainting')
parser.add_argument('--real_dir',  type=str, default= DIR_PROJECT_DATA_RAW)
parser.add_argument('--fake_dir',  type=str, default= DIR_PROJECT_DATA_INPAINTED_RESULTS)
parser.add_argument("--metric", type=str, nargs="+")
args = parser.parse_args()


def read_img(name_pair): 
    rname, fname = name_pair
    rimg = Image.open(rname).resize((512, 512))
    fimg = Image.open(fname).resize((512, 512))
    return np.array(rimg), np.array(fimg)


def main(num_worker=8):
    real_names = []
    fake_names = []
    #self.image_masked_path = []
    
    for filename in os.listdir(args.fake_dir):
        base_name = os.path.splitext(filename)[0]  
        parts = base_name.split('_')  
        desired_part = parts[-1]
        # print(desired_part)
        if desired_part == 'comp':
            fake_names.append(os.path.join(args.fake_dir, filename))
      
            main_name = '_'.join(os.path.splitext(filename)[0].split('_')[:-3])
            real_img_name = f'{main_name}.jpg'
  
            real_names.append(os.path.join(args.real_dir, real_img_name))
        

    
    print(f'real images: {len(real_names)}, fake images: {len(fake_names)}')
    real_images = []
    fake_images = []
    pool = Pool(num_worker)
    for rimg, fimg in tqdm(pool.imap_unordered(read_img, zip(real_names, fake_names)), total=len(real_names), desc='loading images'):
        real_images.append(rimg)
        fake_images.append(fimg)


    # metrics prepare for image assesments
    metrics = {met: getattr(module_metric, met) for met in args.metric}
    evaluation_scores = {key: 0 for key,val in metrics.items()}
    for key, val in metrics.items():
        evaluation_scores[key] = val(real_images, fake_images, num_worker=num_worker)
    print(' '.join(['{}: {:6f},'.format(key, val) for key,val in evaluation_scores.items()]))
  
  


if __name__ == '__main__':
    main()