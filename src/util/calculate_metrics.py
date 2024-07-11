import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import load_model
import tensorflow as tf
import csv
from util.metrics import dsc, recall, precision, dice_loss, overall_accuracy, jaccard_index
import util.consts as consts

def Calculate_Metrics(y_test, preds, thresh= 0.5):
    dsc_sc = dsc(y_test, preds)
    recall_sc = recall(y_test, preds)
    precision_sc = precision(y_test, preds)
    dsc_loss = dice_loss(y_test, preds)
    accuracy = overall_accuracy(y_test, preds)
    jaccard_sc = jaccard_index(y_test, preds)
    AUC_ROC = roc_auc_score(preds.ravel()>thresh, y_test.ravel())
    return dsc_sc, jaccard_sc, recall_sc, precision_sc, dsc_loss, accuracy, AUC_ROC

def Show_Metrics(dsc_sc, jaccard_sc, recall_sc, precision_sc, dsc_loss, accuracy, AUC_ROC, thresh=0.5):

    print('-'*30)
    print('USING THRESHOLD', thresh)

    print('\n DSC \t{0:5.3f}  \n Jaccard \t{1:5.3f} \n Recall \t{2:5.3f} \n Precision \t{3:5.3f}  \n DSC Loss \t{4:5.3f}'.format(
            dsc_sc,
            jaccard_sc,
            recall_sc,
            precision_sc,
            dsc_loss))
    print(' Global Acc \t{0:^.3f}'.format(accuracy))
    print(' AUC ROC \t{0:^.3f}'.format(AUC_ROC))
    print('\n')


def evaluate_models_segmentation(imgs, masks, model_name, color_space, n_seeds):
    dsc_sc, jacc_sc, rec_sc, prec_sc, dsc_loss, acc, auc = ([] for _ in range(7))

    for idx in range(n_seeds):
        model_path = str(os.path.join(consts.PATH_PROJECT_SEGMENTATION_MODELS, model_name, f'{model_name}-{color_space}-model-{idx}.keras'))

        X_train, X_test, Y_train, Y_test = train_test_split(imgs, masks,  test_size=0.2, random_state=idx)
        X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=idx)

        loaded_model = load_model(model_path, compile=False)

        preds = loaded_model.predict(X_test)
        preds = np.squeeze(preds, axis=-1)

        thresh=0.5

        dsc_sc_curr, jacc_sc_curr, rec_sc_curr, prec_sc_curr, dsc_loss_curr, acc_curr, auc_curr = Calculate_Metrics(Y_test, preds)

        dsc_sc.append(float(tf.reduce_mean(dsc_sc_curr).numpy()))
        jacc_sc.append(float(tf.reduce_mean(jacc_sc_curr).numpy()))
        rec_sc.append(float(tf.reduce_mean(rec_sc_curr).numpy()))
        prec_sc.append(float(tf.reduce_mean(prec_sc_curr).numpy()))
        dsc_loss.append(float(tf.reduce_mean(dsc_loss_curr).numpy()))
        acc.append(float(acc_curr))
        auc.append(float(auc_curr))

    print('MEAN')
    Show_Metrics(np.mean(dsc_sc),np.mean(jacc_sc), np.mean(rec_sc), np.mean(prec_sc), np.mean(dsc_loss), np.mean(acc), np.mean(auc))

    print('STD')
    Show_Metrics(np.std(dsc_sc), np.std(jacc_sc), np.std(rec_sc), np.std(prec_sc), np.std(dsc_loss), np.std(acc), np.std(auc))

    return dsc_sc, jacc_sc, rec_sc, prec_sc, dsc_loss, acc, auc


def save_metrics_to_csv(metrics_dict, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = metrics_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(metrics_dict)


def evaluate_models_segmentation_and_save_metrics(imgs, masks, model_name, color_space, n_seeds):
    metrics_dict = {}

    dsc_sc, jacc_sc, rec_sc, prec_sc, dsc_loss, acc, auc=evaluate_models_segmentation(imgs, masks, model_name, color_space, n_seeds)
    metrics_dict['dsc_mean'] = np.mean(dsc_sc)
    metrics_dict['jaccard_mean'] = np.mean(jacc_sc)
    metrics_dict['rec_mean'] = np.mean(rec_sc)
    metrics_dict['prec_mean'] = np.mean(prec_sc)
    metrics_dict['dsc_loss_mean'] = np.mean(dsc_loss)
    metrics_dict['acc_mean'] = np.mean(acc)
    metrics_dict['auc_mean'] = np.mean(auc)

    metrics_dict['dsc_std'] = np.std(dsc_sc)
    metrics_dict['jaccard_std'] = np.std(jacc_sc)
    metrics_dict['rec_std'] = np.std(rec_sc)
    metrics_dict['prec_std'] = np.std(prec_sc)
    metrics_dict['dsc_loss_std'] = np.std(dsc_loss)
    metrics_dict['acc_std'] = np.std(acc)
    metrics_dict['auc_std'] = np.std(auc)
    
    model_path = str(os.path.join(consts.PATH_PROJECT_SEGMENTATION_METRICS, f'{model_name}-{color_space}-model.csv'))

    save_metrics_to_csv(metrics_dict, model_path)


def evaluate_models_classification_and_save_metrics(recall, specificity, aucroc, model_name, dataset):
    metrics_dict = {}

    metrics_dict['recall_mean'] = np.mean(recall)
    metrics_dict['specificity_mean'] = np.mean(specificity)
    metrics_dict['aucroc_mean'] = np.mean(aucroc)
    
    metrics_dict['recall_std'] = np.std(recall)
    metrics_dict['specificity_std'] = np.std(specificity)
    metrics_dict['aucroc_std'] = np.std(aucroc)
    
    path = str(os.path.join(consts.PATH_PROJECT_CLASSIFICATION_METRICS, f'{model_name}-{dataset}-metrics.csv'))

    save_metrics_to_csv(metrics_dict, path)
