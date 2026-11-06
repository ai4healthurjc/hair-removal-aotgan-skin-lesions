# hair-removal-aotgan-skin-lesions

Repository for the reproducibility of the results of the paper *"A data-driven approach for digital hair removal in dermoscopy images using encoder-decoder and generative adversarial network-based models"*. The proposed method removes hair from dermoscopy images in two stages: hair segmentation with U-Net-based models (Dense-Net, Att-Net, and Att+Dense-Net) and hair inpainting with an aggregated contextual-transformation generative adversarial network (AOT-GAN), at 512 × 512 pixels.

No images, masks, metadata, or trained models are distributed in this repository. All the datasets must be downloaded from their original sources (see below).

## Installation and setup

Clone the repository:

```console
git clone https://github.com/ai4healthurjc/hair-removal-aotgan-skin-lesions.git
cd hair-removal-aotgan-skin-lesions
```

Create and activate a Python virtual environment (conda or virtualenv) and install the libraries:

```console
pip install -r requirements.txt
```

The experiments were run with Python 3, TensorFlow 2.13.1 and Keras 2.13.1 (segmentation and classification), keras-unet-collection 0.1.13 (U-Net-based architectures), and PyTorch 2.3.1 with torchvision 0.18.1, CUDA 12.1 and cuDNN 8.9.2 (AOT-GAN). The complete list of dependencies and their versions is provided in `requirements.txt`.

Hardware used to run the experiments (a single GPU was used for each experiment):

- CPU: AMD EPYC 7713, 64 cores, 128 threads, 2.0 GHz
- RAM: 256 GB DDR4 (8 x 32 GB)
- GPUs: 3 x NVIDIA RTX 4000 Ada (20 GB GDDR6)
- Storage: 1 TB NVMe PCIe 4.0 (system drive), 4 TB NVMe PCIe 4.0 (data storage)

## Datasets

| Dataset | Use in the paper | Source |
|---|---|---|
| CA | Hair segmentation (training and test) | [Mendeley Data](https://data.mendeley.com/datasets/j5ywpd2p27/2) |
| AGH (Skin_Hair dataset) | Hair inpainting (training and test) and cross-dataset evaluation of segmentation | [SHD](https://skin-hairdataset.github.io/SHD/) |
| PH2 | External evaluation set with synthetic hair, melanoma classification, and expert assessment | [PH2 database](https://www.fc.up.pt/addi/ph2%20database.html) |
| Derm7pt | Melanoma classification and expert assessment | [Derm7pt](https://derm.cs.sfu.ca/Welcome.html) |
| ISIC-2020 | Melanoma classification | [ISIC 2020 Challenge](https://challenge2020.isic-archive.com/) |

After downloading the datasets, place them in the `data` folder:

- `data/segmentation_hair/raw` (images) and `data/segmentation_hair/masks` (hair masks): CA dataset.
- `data/inpainting/raw_without_hair` (hair-free images), `data/inpainting/raw_with_hair` (images with hair), and `data/inpainting/masks` (hair masks): AGH dataset.
- `data/classification/ISIC-2020`, `data/classification/Derm7pt`, and `data/classification/PH2`: images and the metadata file (CSV) provided by each dataset, with the diagnosis of each image (`target` in ISIC-2020, `diagnosis` in Derm7pt, and `Clinical Diagnosis` in PH2, with 1 for melanoma and 0 otherwise). In ISIC-2020, the duplicated images and the images with an unknown diagnosis were discarded, as described in the Supplementary Material of the paper.

## Data partitions

All experiments were repeated with five train-validation-test partitions generated with fixed random seeds (0 to 4). Each dataset was split into training (80%) and test (20%) subsets, and 15% of the training subset was used for validation. In the AGH dataset, the split was performed at the level of the original hair-free images, so that all the hair-augmented versions of the same image belong to the same subset. In the classification datasets, the split was stratified by class, and the majority class was randomly undersampled only in the training subset.

The identifiers of the images in each subset and partition are provided in the folder `partitions`:

- `segmentation_CA_partitions.csv`
- `inpainting_AGH_partitions.csv`
- `classification_Derm7pt_partitions.csv`, `classification_ISIC-2020_partitions.csv`, and `classification_PH2_partitions.csv`
- `ph2_hairfree_ids.txt` (the 21 hair-free PH2 images used to build the external evaluation set)

These files can be regenerated from the downloaded datasets with:

```console
python src/export_partitions.py --experiment all
```

## External PH2 evaluation set with synthetic hair

Since the terms of use of PH2 do not allow its redistribution, the external evaluation set is not distributed. It can be generated exactly (pixel-identical images) from the original PH2 images with `src/make_ph2_external_set.py`, in two steps:

1. Read the 21 hair-free PH2 images listed in `partitions/ph2_hairfree_ids.txt` from the original PH2 folder and resize them to 512 × 512 pixels:

```console
python src/make_ph2_external_set.py prepare [PH2 folder] partitions/ph2_hairfree_ids.txt data/ph2_external_set/ph2_original
```

2. Add synthetic hair with a procedural hair simulator (3 hair densities × 3 hair colors), which generates the 189 images with hair, their binary hair masks, and a manifest (`manifest.csv`):

```console
python src/make_ph2_external_set.py synthesize data/ph2_external_set/ph2_original data/ph2_external_set 0
```

With seed 0, the script reproduces exactly the set used in the paper (on average, hair covers 3.3%, 4.4%, and 6.6% of the image for low, medium, and high density).

## Hair segmentation

To train the U-Net-based models (`--model` = `Dense-U-Net`, `Attention-U-Net`, or `Attention-Dense-U-Net`; `--color_space` = `rgb` or `ycrcb`) on the five partitions:

```console
python src/segmentation_train.py --model=Attention-U-Net --resize=True --img_size=512 --color_space=ycrcb --n_seeds=5 --epochs_early_stopping=55 --epochs_lr_adaptative=25 --batch_size=16 --num_epochs=250
```

To generate the hair masks with a trained model:

```console
python src/generate_masks_hair.py --model=Attention-U-Net --resize=True --img_size=512 --color_space=ycrcb --seed=0
```

The hair masks are saved in `reports/segmentation/masks`.

## Hair inpainting with AOT-GAN

To train AOT-GAN on one partition (`--train_seed` = 0 to 4):

```console
python src/AOT_GAN/train.py --image_size=512 --iterations=1e4 --save_every=2e3 --batch_size=8 --train_seed=0 --dir_image=data/inpainting/raw_without_hair --dir_mask=data/inpainting/masks
```

The loss weights are λ_adv = 0.01, λ_rec = 1, λ_per = 0.1, and λ_style = 250 (default values in `src/AOT_GAN/my_utils/option.py`). The models are saved in `reports/inpainting/aot_gan/models`.

To remove the hair of the test images with a trained model, using the hair masks detected by Att-Net:

```console
python src/AOT_GAN/test.py --pre_train=[path to the trained model] --dir_image_test=data/inpainting/raw_with_hair --dir_mask=[path to the hair masks]
```

The hair-free images are saved in `reports/inpainting/aot_gan`.

To evaluate the hair-free images with respect to the ground truth:

```console
python src/AOT_GAN/eval.py --real_dir=data/inpainting/raw_without_hair --fake_dir=reports/inpainting/aot_gan --metric=ssim
```

## Traditional hair removal methods

To remove the hair with DullRazor or Line-filtering (`--traditional_method` = `DullRazor` or `Huang`):

```console
python src/main_inpainting_traditional_methods.py --traditional_method=DullRazor --resize=True --img_size=512 --color_space=rgb
```

The hair-free images are saved in `reports/inpainting/traditional_methods`.

## Melanoma classification

To train the CNN-based classifiers (`--model` = `Resnet`, `mobilenet`, or `vgg`; `--dataset` = `ISIC-2020`, `Derm7pt`, or `PH2`) on the five partitions:

```console
python src/classification.py --dataset=ISIC-2020 --model=Resnet --n_seeds=5 --epochs_early_stopping=15 --epochs_lr_adaptative=5 --batch_size=64 --num_epochs=200
```

## Citation

If you use this code, please cite the paper (citation details will be added upon publication).
