__author__ = 'Paul F. Jaeger'

import configs as cf
from data_loader import load_NCI_ISBI_dataset, create_data_gen_pipeline
import tensorflow as tf
from model import create_nice_UNet as create_UNet
from utils import numpy_dice_per_class, get_one_hot_prediction
from plotting import plot_batch_prediction
import numpy as np
import os


# define graph
tf.reset_default_graph()

x = tf.placeholder('float', shape=[None, cf.patch_size[0], cf.patch_size[0], cf.n_channels])
logits, variables = create_UNet(x, features_root=cf.features_root, n_classes=cf.n_classes, is_training=False)
predicter = tf.nn.softmax(logits)

# initialize
saver = tf.train.Saver()
test_data = load_NCI_ISBI_dataset(cf, split='test')

with tf.Session() as sess:
    fold = 0
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(cf.exp_dir, 'params_{}'.format(fold)))

    for ix, pid in enumerate(test_data['pid']):

        test_gen = create_data_gen_pipeline(test_data, cf=cf, test_ix=ix, do_aug=False)
        te_patient = next(test_gen)
        correct_prediction = np.argmax(sess.run(predicter, feed_dict={x: te_patient['data']}), axis=3)
        print "DICES", numpy_dice_per_class(get_one_hot_prediction(correct_prediction, cf.n_classes), te_patient['seg']) #is this 3D properly?
        plot_batch_prediction(te_patient, correct_prediction, cf.n_classes, os.path.join(cf.exp_dir, '{}_pred.png'.format(pid)))

