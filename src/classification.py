import os
import cv2
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn import metrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from util.loader import load_images_classification, load_model_classification
import util.consts as consts
from util.calculate_metrics import evaluate_models_classification_and_save_metrics

import logging
import coloredlogs
import argparse

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    
    parser.add_argument('--dataset', default='ISIC-2020', type=str)
    parser.add_argument('--model', default='U-Net', type=str)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--epochs_early_stopping', default=55, type=int)
    parser.add_argument('--epochs_lr_adaptative', default=25, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=600, type=int)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='melanoma classification in skin lesions images')
args = parse_arguments(parser)

X, Y=load_images_classification(args.dataset)

recall=[]
specificity=[]
auc=[]

for idx in range (args.n_seeds):

    logger.info('Training with {}: partition: {}'.format(args.model, idx))

    model=load_model_classification (args.model, args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=idx)
    
    rus = RandomUnderSampler(sampling_strategy='majority',  random_state=idx)

    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    X_train_=[]
    X_test_=[]

    for i in range(len(X_train_resampled)):
        X_train_.append(cv2.resize(cv2.imread(X_train_resampled[i][0]), (224,224)))

    for i in range(len(X_test)):
        X_test_.append(cv2.resize(cv2.imread(X_test[i][0]), (224,224)))

    X_train_res, X_val, y_train_res, y_val = train_test_split(X_train_, y_train_resampled, stratify=y_train_resampled, test_size=0.15, random_state=idx)

    X_train=np.array(X_train_res)
    X_val=np.array(X_val)
    X_test=np.array(X_test_)

    Y_train = to_categorical(y_train_res, num_classes= 2)
    Y_val=to_categorical(y_val, num_classes= 2)
    Y_test = to_categorical(y_test, num_classes= 2)
    
    
    model_path = str(os.path.join(consts.PATH_PROJECT_CLASSIFICATION_MODELS, args.model, f'{args.model}-{args.dataset}-model-{idx}.h5'))

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    learn_control = ReduceLROnPlateau(monitor='val_loss', patience=args.epochs_lr_adaptative,
                                    verbose=1,factor=0.2, min_lr=1e-7)

    early = EarlyStopping(monitor="val_loss", 
                        mode="min", 
                        patience=args.epochs_early_stopping) 
    
    train_generator = ImageDataGenerator(
        zoom_range=2, 
        rotation_range = 90,
        horizontal_flip=True, 
        vertical_flip=True, 
    )
    history = model.fit_generator(
    train_generator.flow(X_train, Y_train, batch_size=args.batch_size),
    steps_per_epoch=int(round(X_train.shape[0] / args.batch_size)),
    epochs=args.num_epochs,
    validation_data=(X_val, Y_val),
    callbacks=[learn_control, early, checkpoint])
    
    hist_df = pd.DataFrame(history.history) 
    csv_dir = os.path.join(consts.PATH_PROJECT_CLASSIFICATION_HIST, args.model)

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    csv_path = os.path.join(csv_dir, f'hist_{args.model}-{args.dataset}-model-{idx}.csv')
    hist_df.to_csv(csv_path, index=False)

    loaded_model = load_model(model_path, compile=False)

    Y_pred = model.predict(X_test)
    Recall=recall_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
    tn, fp, fn, tp = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1)).ravel()
    Specificity = tn / (tn + fp)
    fpr, tpr, threshold = metrics.roc_curve(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
    roc_auc = metrics.auc(fpr, tpr)
    
    recall.append(Recall)
    specificity.append(Specificity)
    auc.append(roc_auc)

evaluate_models_classification_and_save_metrics(recall, specificity, auc, args.model, args.dataset)


