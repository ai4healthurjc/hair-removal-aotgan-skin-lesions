import os
from keras.callbacks import EarlyStopping
import math
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from util.loader import load_data_segmentation, load_unet_model
import util.consts as consts
from util.metrics import dsc, dice_loss
from util.calculate_metrics import evaluate_models_segmentation_and_save_metrics

import logging
import coloredlogs
import argparse
import math

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--model', default='U-Net', type=str)
    parser.add_argument('--resize', default=True, type=bool)
    parser.add_argument('--img_size', default=128, type=int)
    # parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--color_space', default='ycrcb', type=str)
    parser.add_argument('--num_channel', default=1, type=int)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--epochs_early_stopping', default=55, type=int)
    parser.add_argument('--epochs_lr_adaptative', default=25, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=600, type=int)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='hair segmentation in skin lesions images')
args = parse_arguments(parser)

imgs, masks, input_size, img_names, mask_names=load_data_segmentation(resize=args.resize, img_row=args.img_size, img_col=args.img_size, color_spc=args.color_space)


for idx in range (args.n_seeds):

    logger.info('Training with {}, color_space: {}, partition: {}'.format(args.model,
                                                                           args.color_space,
                                                                           idx)
                    )

    model=load_unet_model (args.model, args.num_channel, input_size)
    model.compile(loss=dice_loss,optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=[dsc])
    model_path = str(os.path.join(consts.PATH_PROJECT_SEGMENTATION_MODELS, args.model, f'{args.model}-{args.color_space}-model-{idx}.h5'))
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    learn_control = ReduceLROnPlateau(monitor='val_loss', patience=args.epochs_lr_adaptative, verbose=1, factor=0.2, min_lr=1e-15)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=args.epochs_early_stopping)


    X_train, X_test, Y_train, Y_test = train_test_split(imgs, masks,  test_size=0.2, random_state=idx)
    X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=idx)

    hist = model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size=args.batch_size, epochs=args.num_epochs, verbose=1, steps_per_epoch=math.ceil(X_train.shape[0] / args.batch_size), callbacks=[learn_control, early, checkpoint])
    hist_df = pd.DataFrame(hist.history)
    csv_dir = os.path.join(consts.PATH_PROJECT_SEGMENTATION_HIST, args.model)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, f'hist_{args.model}-{args.color_space}-model-{idx}.csv')
    hist_df.to_csv(csv_path, index=False)

evaluate_models_segmentation_and_save_metrics(imgs, masks, args.model, args.color_space, args.n_seeds)
