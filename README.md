# hair-removal-aotgan-skin-lesions
Repository for reproducibility results for the paper 'A data-driven hair removal approach for dermoscopy images through modified U-Net and GAN-based models', which is focused on eliminating hair from dermoscopy images of skin lesions by employing various U-Net architectures for hair segmentation, followed by the application of an optimized GAN model, the Aggregated Contextual-Transformation-GAN, for image inpainting.

## Installation and setup

To download the source code, you can clone it from the Github repository.
```console
git clone git@github.com:ai4healthurjc/hair-removal-aotgan-skin-lesions.git
```

Before installing libraries, ensuring that a Python virtual environment is activated (using conda o virtualenv). To install Python libraries run: 

```console
pip install -r requirements.txt
```
## Download dermoscopy images for hair segmentation and hair inpainting from public datasets

Datasets with dermoscopy images are publicly available in the following websites:

1. [Link to CA dataset](https://data.mendeley.com/datasets/j5ywpd2p27/2) (for hair segmentation)
2. [Link to AGH dataset](https://skin-hairdataset.github.io/SHD/) (for hair inpainting)

## Download dermoscopy images for melanoma classification from public datasets

Datasets with dermoscopy images are publicly available in the following websites:

1. [Link to ISIC20 dataset](https://challenge2020.isic-archive.com/)
2. [Link to PH2 dataset](https://www.fc.up.pt/addi/ph2%20database.html)
3. [Link to Derm7pt dataset](https://derm.cs.sfu.ca/Welcome.html)

To replicate the results, download images from datasets. After downloading data, you have to put images in data folder. Specifically:

- data/segmentation_hair/masks (hair masks), data/segmentation_hair/raw (images). (CA dataset for hair segmentation);
- data/inpainting/raw_with_hair (images with hair), data/inpainting/raw_without_hair (images without hair), data/inpainting/masks (hair masks). (AGH dataset for hair inpainting);
- data/classification/ISIC-2020 (metadata and images);
- data/classification/Derm7pt (metadata and images);
- data/classification/PH2 (metadata and images);

It is important to note that, for hair inpainting, we divided the dataset into five different train-test sets. This means that when you execute the procedure, you need to be aware that the images without hair and masks folders were updated with different training sets. When testing, we used the different test sets accordingly.

## To obtain different results of data-driven models

To train U-Net-based model for hair segmentation:
    
```console
python src/segmentation_train.py --model=Attention-U-Net --resize=True --img_size=128 --color_space=ycrcb --n_seeds=5 --epochs_early_stopping=55 --epochs_lr_adaptative=25 --batch_size=32 --num_epochs=500
```

To generate hair masks with U-Net-based models trained:

```console
python src/generate_masks_hair.py --model=Attention-U-Net --resize=True --img_size=128 --color_space=ycrcb --seed=0 
```
The hair masks are saved in: reports/segmentation/masks.

To train AOT-GAN model for hair inpainting:

```console
 python src/AOT_GAN/train.py --image_size=512 --iterations=1e4 --save_every=2e3 --batch_size=8 --dir_image=[path to dermoscopy image without hair (ground truths)] --dir_mask=[path to hair masks]
```
The AOT-GAN models are saved in: reports/inpainting/aot_gan/models

By default, the following paths are set:
```console
--dir_image=data/inpainting/raw_without_hair
--dir_mask=data/inpainting/masks 
```

To generate dermoscopy images without hair with AOT-GAN-based model trained:

```console
python src/AOT_GAN/test.py --pre_train=[path to pretrained model] --dir_image_test= [path to dermoscopy images with hair] --dir_mask=[path to hair masks]
```
The dermoscopy images without hair are saved in: reports/inpainting/aot_gan

By default, the following paths are set:
```console
--dir_image_test=data/inpainting/raw_with_hair
--dir_mask=data/inpainting/masks 
```

To evaluate the dermoscopy images without hair by AOT-GAN with real dermoscopy images images without hair (ground truth):

```console
 python src/AOT_GAN/eval.py --real_dir [real dermoscopy images without hair (ground truths)] --fake_dir [inpainting results by AOT-GAN] --metric=ssim [mae fid psnr]
```
By default, the following paths are set:
```console
--real_dir=data/inpainting/raw_without_hair
--fake_dir=reports/inpainting/aot_gan
```

To generate dermoscopy images without hair with traditional inpainting methods:

```console
python src/main_inpainting_traditional_methods.py --traditional_method=DullRazor --resize=True --img_size=512 --color_space=rgb
```
The dermoscopy images without hair are saved in: reports\inpainting\traditional_methods\DullRazor or reports\inpainting\traditional_methods\Huang

By default, it takes the masks located in the path data/inpainting/masks and the images with hair from data/inpainting/raw_with_hair

To train CNN-based models for melanoma classification:

```console
python src/classification.py --dataset=ISIC-2020 --model=Resnet --n_seeds=5 --epochs_early_stopping=15 --epochs_lr_adaptative=5 --batch_size=64 --num_epochs=200
```



