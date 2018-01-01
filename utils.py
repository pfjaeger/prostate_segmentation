import tensorflow as tf
import numpy as np
import logging
import inspect
import os


def _get_loss(logits, y, n_classes, loss_name, class_weights=None, dim=2):
	"""
	Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
	Optional arguments are:
	class_weights: weights for the different classes in case of multi-class imbalance
	regularizer: power of the L2 regularizers added to the loss function
	"""

	flat_logits = tf.reshape(logits, [-1, n_classes])
	flat_labels = tf.reshape(y, [-1, n_classes])

	if loss_name == 'weighted_cross_entropy':
		weight_map = tf.reduce_sum(tf.multiply(flat_labels, class_weights), axis=1)
		loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
		loss = tf.reduce_mean(tf.multiply(loss_map, weight_map))

	elif loss_name == 'cross_entropy':
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))

	elif loss_name == 'dice_coefficient':
		loss= 1. - tf.reduce_mean(get_dice_per_class(logits, y, dim))

	return loss



def get_dice_per_class(logits, y, dim=3):
	#in 3D this gives the dice over 3 dims.
	#in 2D this  gives the dice over 2dims + batch.
	eps = tf.constant(float(1e-6))
	axes = tuple(range(dim - 2, dim + 1)) # batch as pseudo volume in 2D
	prediction = tf.nn.softmax(logits)
	intersection = tf.reduce_sum(prediction * y, axis=axes)
	union = eps + tf.reduce_sum(prediction, axis=axes) + tf.reduce_sum(y, axis=axes)
	dice_per_class = tf.constant(2.) * intersection / union
	if dim == 3:
		dice_per_class = tf.reduce_mean(dice_per_class, axis=0)
	return dice_per_class



def numpy_volume_dice_per_class(prediction, y):

	eps = 1e-4
	axes = tuple(range(0, len(y.shape) - 1))
	intersection = np.sum(prediction * y, axis=axes)
	union = eps + np.sum(prediction, axis=axes) + np.sum(y, axis=axes)
	dice_per_class = 2 * intersection / union
	return dice_per_class



def get_class_weights(seg, margin=0.1):
	"""
	get class weight values for a vector of pixels with sh....
	return weight vect...
	"""
	spatial_axes = tuple(range(1, len(seg.shape)-1))
	class_counts = np.sum(seg, axis=spatial_axes)
	spatial_counts = np.prod(np.array([seg.shape[ix] for ix in spatial_axes]))
	class_weights = 1 - (class_counts / float(spatial_counts)) + margin		 ###types???
	class_weights_flat = np.repeat(class_weights, seg.shape[1] ** len(spatial_axes), axis=0)
	return class_weights_flat



def get_one_hot_prediction(pred, n_classes):
	"""
	transform a softmax prediction to a one-hot prediction of the same shape
	"""
	pred_one_hot = np.zeros(list(pred.shape) + [n_classes]).astype('int32')
	for cl in range(n_classes):
		pred_one_hot[..., cl][pred == cl] = 1
	return pred_one_hot



def get_logger(cf):


	logger = logging.getLogger('UNet_training')
	log_file = cf.exp_dir + '/exec.log'
	print('Logging to {}'.format(log_file))
	hdlr = logging.FileHandler(log_file)
	logger.addHandler(hdlr)
	# logger.addHandler(logging.StreamHandler())
	logger.setLevel(logging.DEBUG)
	logger.info('Created Exp. Dir: {}.'.format(cf.exp_dir))
	return logger


def prep_exp(cf):

	for dir in [cf.exp_dir, cf.test_dir, cf.plot_dir]:
		if not os.path.exists(dir):
			os.mkdir(dir)

	lines = inspect.getsourcelines(cf)
	file = open(os.path.join(cf.exp_dir, 'configs.py'), 'w')
	file.write("".join(lines[0]))
	file.close()