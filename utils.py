__author__ = 'Paul F. Jaeger'

import tensorflow as tf
import numpy as np
import logging
import inspect
import os


def _get_loss(logits, y, n_classes, loss_name, class_weights=None, dim=2):
    """
    construct the loss function, either cross_entropy, weighted_cross_entropy or dice_coefficient.
    :param logits: network output tensor. shape [b, x, y, n_classes] (2D) / [b, z, x, y, n_classes] (3D)
    :param y: target tensor. shape [b, x, y, n_classes] (2D) / [b, z, x, y, n_classes] (3D)
    :param n_classes: number of classes used to flatten the logits in case of weighted_cross_entropy
    :param loss_name: 'cross_entropy', 'weighted_cross_entropy' or 'dice_coefficient'
    :param class_weights: (optional)  weights for the different classes in case of multi-class imbalance
    :param dim: (optional) 2D or 3D training needs to be specified if using dice_coefficient
    :return: loss function.
    """

    flat_logits = tf.reshape(logits, [-1, n_classes])
    flat_labels = tf.reshape(y, [-1, n_classes])
    if loss_name == 'weighted_cross_entropy':
        class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        weight_map = tf.reduce_sum(tf.multiply(flat_labels, class_weights), axis=1)
        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
        loss = tf.reduce_mean(tf.multiply(loss_map, weight_map))
    elif loss_name == 'cross_entropy':
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))
    elif loss_name == 'dice_coefficient':
        loss = 1. - tf.reduce_mean(get_dice_per_class(logits, y, dim))
    else:
        raise ValueError("wrong loss name specified in configs.")
    return loss


def get_dice_per_class(logits, y, dim):
    """
    dice loss function. For 3D, computes the dice loss per volume and averages over batch_size. For 2D
    uses the batch_size as a pseudo volume dimension, i.e. computes the dice over the entire batch.
    :param logits: network output tensor. shape [b, x, y, n_classes] (2D) / [b, z, x, y, n_classes] (3D)
    :param y: target tensor. shape [b, x, y, n_classes] (2D) / [b, z, x, y, n_classes] (3D)
    :param dim: 2D or 3D network training.
    :return: dice scores per class. shape [n_classes]
    """
    eps = tf.constant(float(1e-6))
    axes = tuple(range(dim - 2, dim + 1))
    prediction = tf.nn.softmax(logits)
    intersection = tf.reduce_sum(prediction * y, axis=axes)
    union = eps + tf.reduce_sum(prediction, axis=axes) + tf.reduce_sum(y, axis=axes)
    dice_per_class = tf.constant(2.) * intersection / union
    if dim == 3:
        dice_per_class = tf.reduce_mean(dice_per_class, axis=0)
    return dice_per_class


def get_slicewise_dice_per_class(logits, y):
    """
    standard 2D dice loss computed over spatial axes and averaged over batch. Only used for baseline method.
    """
    eps = tf.constant(float(1e-6))
    prediction = tf.nn.softmax(logits)
    intersection = tf.reduce_sum(prediction * y, axis=(1, 2))
    union = eps + tf.reduce_sum(prediction, axis=(1, 2)) + tf.reduce_sum(y, axis=(1, 2))
    dice_per_class = tf.reduce_mean(tf.constant(2.) * intersection / union, axis=0)
    return dice_per_class


def numpy_volume_dice_per_class(prediction, y):
    """
    numpy dice loss, used for testing. Computes the dice loss over a given patient volume.
    :param prediction: 3D array containing class predictions. shape [z, x, y]
    :param y: 3D arrray of target classes. shape [z, x, y]
    :return: dice scores per class. shape [n_classes]
    """
    eps = 1e-4
    axes = tuple(range(0, len(y.shape) - 1))
    intersection = np.sum(prediction * y, axis=axes)
    union = eps + np.sum(prediction, axis=axes) + np.sum(y, axis=axes)
    dice_per_class = 2 * intersection / union
    return dice_per_class


def get_one_hot_prediction(prediction, n_classes):
    """
    transform a softmax prediction to a one-hot prediction of the same shape
    :param prediction: 3D array containing class predictions. shape [z, x, y]
    :param n_classes: number of classes
    :return: one-hot encoded prediction. shape: [z, x, y, n_classes]
    """
    pred_one_hot = np.zeros(list(prediction.shape) + [n_classes]).astype('int32')
    for cl in range(n_classes):
        pred_one_hot[..., cl][prediction == cl] = 1
    return pred_one_hot


def get_logger(exp_dir):

    logger = logging.getLogger('UNet_training')
    log_file = exp_dir + '/exec.log'
    print('Logging to {}'.format(log_file))
    hdlr = logging.FileHandler(log_file)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger


def prep_exp(cf):

    for d in [cf.exp_dir, cf.test_dir, cf.plot_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    lines = inspect.getsourcelines(cf)
    f = open(os.path.join(cf.exp_dir, 'configs.py'), 'w')
    f.write("".join(lines[0]))
    f.close()