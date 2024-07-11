import numpy as np
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
import scipy


def ssim_loss(y_true, y_pred):
    ssim_value = tf.image.ssim(y_true * 255, y_pred * 255, max_val=255.0)
    mean_ssim_value = tf.reduce_mean(ssim_value)
    return 1 - mean_ssim_value

def psnr_val(y_true, y_pred, max_value=1.0, smooth=1e-6):
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    if tf.math.less_equal(tf.reduce_max(y_true), 1.0):

        return 20 * tf.math.log(max_value / tf.sqrt(mse + smooth)) / tf.math.log(10.0)
    else:
        return 20 * tf.math.log(255.0 / tf.sqrt(mse + smooth)) / tf.math.log(10.0)

def content_loss(y_true, y_pred):
    return keras.losses.MeanSquaredError()(y_true, y_pred)

def adversarial_loss(y_true, y_pred):
    return -K.mean(K.log(y_pred + 1e-8))

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_fm = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_fm * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_fm) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def iou(y_true, y_pred, smooth=1e-15):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def bce_dice_loss(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def focal_loss(y_true, y_pred):
    alpha=0.25
    gamma=2
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    return tf.reduce_mean(loss)

def recall(y_true, y_pred, smooth=1e-15):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    true_positives = tf.reduce_sum(y_true * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)
    return recall

def precision(y_true, y_pred, smooth=1e-15):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    true_positives = tf.reduce_sum(y_true * y_pred)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
    return precision

def inaccuracy(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    incorrect = tf.reduce_sum(tf.abs(y_true - tf.round(y_pred)))
    total_pixels = tf.cast(tf.size(y_true), tf.float32)
    inaccuracy = incorrect / total_pixels
    return inaccuracy

def overall_accuracy(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    correct_pixels = tf.reduce_sum(tf.cast(tf.equal(tf.round(y_pred), y_true), tf.float32))
    total_pixels = tf.cast(tf.size(y_true), tf.float32)
    accuracy = correct_pixels / total_pixels
    return accuracy

def jaccard_index(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred - y_true * y_pred)
    jaccard = intersection / union
    return K.get_value(jaccard).item()

def vifp_val(ref, dist) -> float:
    sigma_nsq = 2
    eps = 1e-20  
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if scale > 1:
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp_value = num / den

    if np.isnan(vifp_value):
        return 1.0
    else:
        return vifp_value