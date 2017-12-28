import tensorflow as tf
import numpy as np


def _get_loss(logits, y, n_classes, loss_name, class_weights):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """

    flat_logits = tf.reshape(logits, [-1, n_classes])
    flat_labels = tf.reshape(y, [-1, n_classes])
    # USE FLAT ONES TO ACCESS NEW STANDARD SOFTMAX FUNCTIONS TAHT WORK MORE STABLE?
    if loss_name == "cross_entropy":

        if class_weights is not None:

            weight_map = tf.multiply(flat_labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)

            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                               labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)

            loss = tf.reduce_mean(weighted_loss)

        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                          labels=flat_labels))
    elif loss_name == "dice_coefficient":
        loss= tf.constant(1.) - get_dice_per_class(logits, y)[1]

    return loss


def _get_optimizer(loss, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print "CHECK UPDATE OPS:", update_ops
    if update_ops:
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer





def softmax_2d(output_map):
    exponential_map = tf.exp(tf.clip_by_value(output_map,-10, 10))
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)


def get_dice_per_class(logits, y, class_weights=None):
    eps = tf.constant(float(1e-4))
    prediction = softmax_2d(logits)
    intersection = tf.reduce_sum(prediction * y, axis=(1, 2))
    if class_weights is None:
        union = eps + tf.reduce_sum(prediction, axis=(1, 2)) + tf.reduce_sum(y, axis=(1, 2))
    else:
        union = eps + tf.reduce_sum(prediction, axis=(1, 2)) + tf.reduce_sum(y, axis=(1, 2))
    dice_per_class = tf.reduce_mean(tf.constant(2.) * intersection / union, axis=0)
    return dice_per_class


def numpy_dice_per_class(prediction, y):


    eps = 1e-4
    intersection = np.sum(prediction * y, axis=(1,2))
    union = eps + np.sum(prediction, axis=(1,2)) + np.sum(y, axis=(1,2))
    dice_per_class = np.mean(2 * intersection / union, axis=0)
    return dice_per_class


def get_class_weights(seg):
    """
	get class weight values for a vector of pixels with sh....
	return weight vect...
	"""
    class_counts = np.sum(np.mean(seg,axis=0), axis=(0,1))
    class_weights = 1 - (class_counts / float(seg.shape[1] ** 2))
    return class_weights



def get_one_hot_prediction(pred, n_classes):
    """
    transform a softmax prediction to a one-hot prediction of the same shape
    """
    pred_one_hot = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], n_classes)).astype('int32')
    for cl in range(n_classes):
        pred_one_hot[:, :, :, cl][pred == cl] = 1

    return pred_one_hot

