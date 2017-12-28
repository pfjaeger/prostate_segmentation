__author__ = 'Paul F. Jaeger'

import configs as cf
import data_loader
import tensorflow as tf
from model import create_nice_UNet as create_UNet
from utils import get_dice_per_class, get_class_weights, _get_loss, _get_optimizer
from plotting import TrainingPlot_2Panel, plot_batch_prediction
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int)
fold = parser.parse_args().fold
if fold is None:
    fold = 0

# define graph
tf.reset_default_graph()

x = tf.placeholder('float', shape=[None, cf.patch_size[0], cf.patch_size[0], cf.n_channels])
y = tf.placeholder('float', shape=[None, cf.patch_size[0], cf.patch_size[0], cf.n_classes])
is_training = tf.placeholder(tf.bool, name='is_training')
class_weights = tf.placeholder('float')
learning_rate = tf.Variable(cf.learning_rate)

logits, variables = create_UNet(x, features_root=cf.features_root, n_classes=cf.n_classes, is_training=is_training)
loss = _get_loss(logits, y, cf.n_classes, cf.loss_name, class_weights)
predicter = tf.nn.softmax(logits)
dice_per_class = get_dice_per_class(logits, y)

# set up training

metrics = {}
metrics['train'] = {'loss': [0.], 'dices': np.zeros(shape=(1, cf.n_classes))}
metrics['val'] = {'loss': [0.], 'dices': np.zeros(shape=(1, cf.n_classes))}
best_metrics = {'loss': [10, 0], 'dices': np.zeros(shape=(cf.n_classes, 2))}
file_name = cf.exp_dir + '/monitor_{}.png'.format(fold)
TrainingPlot = TrainingPlot_2Panel(cf.n_epochs, file_name, cf.experiment_name,
                                   cf.class_dict)

# initialize
optimizer = _get_optimizer(loss, learning_rate=learning_rate)
saver = tf.train.Saver()

# load data
batch_gen = data_loader.get_data_generators(cf, fold)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(cf.n_epochs):

        val_loss_running_mean = 0.
        val_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
        for _ in range(cf.n_val_batches):
            batch = next(batch_gen['val'])
            val_loss, val_dices, cw = sess.run(
                (loss, dice_per_class, class_weights),
                feed_dict={x: batch['data'],
                           y: batch['seg'],
                           is_training: False,
                           class_weights: get_class_weights(batch['seg'])})

            print "CHECK CLASS WEIGHTS", cw
            val_loss_running_mean += val_loss / cf.n_val_batches
            val_dices_running_batch_mean[0] += val_dices / cf.n_val_batches

        metrics['val']['loss'].append(val_loss_running_mean)
        metrics['val']['dices'] = np.append(metrics['val']['dices'], val_dices_running_batch_mean, axis=0)

        train_loss_running_mean = 0.
        train_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
        for _ in range(cf.n_train_batches):
            batch = next(batch_gen['train'])
            train_loss, train_dices, _ = sess.run((loss, dice_per_class, optimizer),
                                                  feed_dict={x: batch['data'],
                                                             y: batch['seg'],
                                                             is_training: True,
                                                             class_weights: get_class_weights(batch['seg'])})

            print ("LOSS", train_loss, epoch)
            train_loss_running_mean += train_loss / cf.n_train_batches
            train_dices_running_batch_mean += train_dices / cf.n_train_batches

        metrics['train']['loss'].append(train_loss_running_mean)
        metrics['train']['dices'] = np.append(metrics['train']['dices'], train_dices_running_batch_mean,
                                              axis=0)

        epoch += 1

        val_loss = metrics['val']['loss'][-1]
        val_dices = metrics['val']['dices'][-1]

        if val_loss < best_metrics['loss'][0]:
            best_metrics['loss'][0] = val_loss
            best_metrics['loss'][1] = epoch
            saver.save(sess, os.path.join(cf.exp_dir, 'params_{}'.format(fold)))

        for cl in range(cf.n_classes):
            if val_dices[cl] > best_metrics['dices'][cl][0]:
                best_metrics['dices'][cl][0] = val_dices[cl]
                best_metrics['dices'][cl][1] = epoch

        TrainingPlot.update_and_save(metrics, best_metrics)

        # plotting example predictions
        batch = next(batch_gen['val'])
        soft_prediction = sess.run((predicter),
                                   feed_dict={x: batch['data'], is_training: False})
        correct_prediction = np.argmax(soft_prediction, axis=3)
        outfile = cf.plot_dir + '/pred_examle_{}.png'.format(0)  ## FOLD!
        plot_batch_prediction(batch, correct_prediction, cf.n_classes, outfile)







