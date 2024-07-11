
from tensorflow.keras.applications import MobileNetV2
from keras.applications import VGG16
from keras.applications import ResNet50
import numpy as np
import pandas as pd
import cv2
import re
import os

from keras.optimizers import *
from keras_unet_collection._model_att_unet_2d import att_unet_2d
from keras_unet_collection import models
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.models import load_model


import util.consts as consts
from util.models_unet import *
from util.traditional_inpainting_methods import DullRazor, HairRemovMed_hyadamhuang

def load_data_segmentation(resize=True, img_row=128, img_col=128, color_spc='ycbcr', allowed_extensions=('png', 'jpg', 'jpeg')):
    images_orig = [] 
    masks_contour = []  

    img_names = []  
    mask_names = [] 

    img_path = consts.PATH_PROJECT_DATA_HR_RAW
    mask_path = consts.PATH_PROJECT_DATA_HR_MASKS

    img_files = sorted([f for f in os.listdir(img_path) if f.endswith(allowed_extensions)])
    mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith(allowed_extensions)])

    for img_file, mask_file in zip(img_files, mask_files):
        img_names.append(img_file)
        mask_names.append(mask_file)


        img = cv2.imread(os.path.join(img_path, img_file))
        if resize:
            img = cv2.resize(img, (img_row, img_col))
        if color_spc == 'ycrcb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img.astype(np.float32) / 255.0
        images_orig.append(img)

        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_row, img_col))
        mask = mask.astype(np.float32) / 255.0
        masks_contour.append(mask)

    imgs_orig = np.array(images_orig)
    masks_contour = np.array(masks_contour)
    img_shape = imgs_orig.shape[1:]

    return imgs_orig, masks_contour, img_shape, img_names, mask_names


def load_unet_model (model_name, num_channel, input_size):

    if model_name == 'U-Net':
        model = Network()
    
    if model_name == 'Dense-U-Net':
        model = model= DenseUNet(input_size, channel=num_channel, use_l2_reg=False)

    if model_name == 'Attention-U-Net':
        model = model= att_unet_2d(input_size, filter_num=[64, 128, 256, 512, 1024], n_labels=num_channel, stack_num_down=2, stack_num_up=1, activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', batch_norm=True, pool='max', unpool=False, name='attunet')

    if model_name == 'Attention-Dense-U-Net':
        model = DenseUNet_AttGate(input_size, channel=num_channel)

    if model_name == 'U-Net3+':
        model = models.unet_3plus_2d(input_size, n_labels=num_channel, filter_num_down=[64, 128, 256, 512], 
                             filter_num_skip='auto', filter_num_aggregate='auto', 
                             stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                             batch_norm=True, pool='max', unpool=False, deep_supervision=False, name='unet3plus')

    return model


def load_raw_images (resize=True, img_row=128, img_col=128, color_spc='ycbcr', allowed_extensions=('png', 'jpg', 'jpeg')):
    images_orig = [] 

    img_names = []  

    img_path = consts.PATH_PROJECT_DATA_HR_RAW

    img_files = sorted([f for f in os.listdir(img_path) if f.endswith(allowed_extensions)])

    for img_file in img_files:
        img_names.append(img_file)

        img = cv2.imread(os.path.join(img_path, img_file))
        if resize:
            img = cv2.resize(img, (img_row, img_col))
        if color_spc == 'ycrcb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img.astype(np.float32) / 255.0
        images_orig.append(img)

    imgs_orig = np.array(images_orig)
    img_shape = imgs_orig.shape[1:]

    return imgs_orig, img_shape, img_names
    

def generate_predictions(model_name, images, img_names, idx, color_space='ycbcr'):

    output_dir = consts.PATH_PROJECT_SEGMENTATION_IMAGES
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(consts.PATH_PROJECT_SEGMENTATION_MODELS, model_name, f'{model_name}-{color_space}-model-{idx}.h5')
    loaded_model = load_model(model_path, compile=False)
    predictions = loaded_model.predict(images)
    predictions = (predictions > 0.5).astype(np.uint8)


    for img_name, prediction in zip(img_names, predictions):
        output_path = os.path.join(output_dir, f'{img_name}')
        cv2.imwrite(output_path, prediction)
        print(f'Prediction saved in: {output_path}')



def load_data_traditional_inpainting(resize=True, img_row=128, img_col=128, color_spc='ycbcr', allowed_extensions=('png', 'jpg', 'jpeg')):
    images_orig = [] 
    masks_contour = []  

    img_names = []  
    mask_names = [] 

    img_path = consts.DIR_PROJECT_DATA_RAW_HAIR
    mask_path = consts.DIR_PROJECT_DATA_MASKS

    img_files = sorted([f for f in os.listdir(img_path) if f.endswith(allowed_extensions)])
    mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith(allowed_extensions)])

    for img_file, mask_file in zip(img_files, mask_files):
        img_names.append(img_file)
        mask_names.append(mask_file)

        img = cv2.imread(os.path.join(img_path, img_file))
        if resize:
            img = cv2.resize(img, (img_row, img_col))
        if color_spc == 'ycrcb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        images_orig.append(img)

        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_row, img_col))
        masks_contour.append(mask)

    imgs_orig = np.array(images_orig)
    masks_contour = np.array(masks_contour)
    img_shape = imgs_orig.shape[1:]

    return imgs_orig, masks_contour, img_shape, img_names, mask_names


def load_traditional_inpainting_method (traditional_method_name, imgs, masks, imgs_names):

    if traditional_method_name == 'DullRazor':
        for img, mask, img_name in zip(imgs, masks, imgs_names):
            inpainted_image = DullRazor(img, mask)
            output_dir = os.path.join(consts.PATH_PROJECT_TRADITIONAL_INPAINTING_IMAGES, traditional_method_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{img_name}')
            cv2.imwrite(output_path, inpainted_image)

    if traditional_method_name == 'Huang':
        for img, mask, img_name in zip(imgs, masks, imgs_names):
            inpainted_image = HairRemovMed_hyadamhuang(img, mask, s=5)
            output_dir = os.path.join(consts.PATH_PROJECT_TRADITIONAL_INPAINTING_IMAGES, traditional_method_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{img_name}')
            cv2.imwrite(output_path, inpainted_image)

def load_images_classification (dataset_name):

    if dataset_name=='ISIC-2020':
        path_meta=consts.PATH_PROJECT_DATA_ISIC
        label='target'
        img_name='image_name'
    elif dataset_name=='Derm7pt':
        path_meta=consts.PATH_PROJECT_DATA_DERM7PT
        label='diagnosis'
    elif dataset_name=='PH2':
        path_meta=consts.PATH_PROJECT_DATA_PH2
        label='Clinical Diagnosis'

    images=[]

    path_meta = str(path_meta)

    def extract_number(file_path):
        match = re.search(r'IMD(\d+)\.(jpg|jpeg|png|bmp)', file_path, re.IGNORECASE)
        return int(match.group(1)) if match else -1
        

    for root, dirs, files in os.walk(path_meta):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                images.append(filepath)
            elif filepath.lower().endswith(".csv"):
                metadata = pd.read_csv(filepath, delimiter=',')
                if dataset_name=='ISIC-2020':
                    metadata= metadata.sort_values(img_name)

    if dataset_name=='PH2':
        images = sorted(images, key=extract_number)

    X = np.array(images).reshape(-1,1) 
    Y= metadata[label]

    return X, Y

def load_model_classification(model_name, dataset_name):

    if model_name == 'Resnet':
        base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3))

        model = build_model(base_model ,lr = 1e-4, starting_layer_name='conv5_block3_3_conv')
        
        
    elif model_name=='vgg':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        model = build_model(base_model ,lr = 1e-4, starting_layer_name='block5_conv3')

    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        model = build_model(base_model ,lr = 1e-4, starting_layer_name='Conv_1')

    return model

