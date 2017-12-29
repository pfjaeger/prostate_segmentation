__author__ = 'Paul F. Jaeger'

import configs as cf
import data_loader
import utils
import tensorflow as tf
from model import create_nice_UNet as create_UNet
from plotting import TrainingPlot_2Panel, plot_batch_prediction
import numpy as np
import argparse
import os
import shutil
import imp

#DISCLAIMER


def train(fold):

    # build tf graph
    tf.reset_default_graph()
    x = tf.placeholder('float', shape=[None, cf.patch_size[0], cf.patch_size[0], cf.n_channels])
    y = tf.placeholder('float', shape=[None, cf.patch_size[0], cf.patch_size[0], cf.n_classes])
    is_training = tf.placeholder(tf.bool, name='is_training')
    class_weights = tf.placeholder('float')
    learning_rate = tf.Variable(cf.learning_rate)
    logits, variables = create_UNet(x, features_root=cf.features_root, n_classes=cf.n_classes, is_training=is_training)
    loss = utils._get_loss(logits, y, cf.n_classes, cf.loss_name, class_weights)
    predicter = tf.nn.softmax(logits)
    dice_per_class = utils.get_dice_per_class(logits, y)

    # set up training
    metrics = {}
    metrics['train'] = {'loss': [0.], 'dices': np.zeros(shape=(1, cf.n_classes))}
    metrics['val'] = {'loss': [0.], 'dices': np.zeros(shape=(1, cf.n_classes))}
    best_metrics = {'loss': [10, 0], 'dices': np.zeros(shape=(cf.n_classes + 1, 2))}
    file_name = cf.exp_dir + '/monitor_{}.png'.format(fold)
    TrainingPlot = TrainingPlot_2Panel(cf.n_epochs, file_name, cf.experiment_name,
                                       cf.class_dict)

    # initialize
    optimizer = utils._get_optimizer(loss, learning_rate=learning_rate)
    saver = tf.train.Saver()
    batch_gen = data_loader.get_train_generators(cf, fold)

    # training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(cf.n_epochs):

            # perform validation
            val_loss_running_mean = 0.
            val_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
            for _ in range(cf.n_val_batches):
                batch = next(batch_gen['val'])
                cw = utils.get_class_weights(batch['seg'])
                val_loss, val_dices = sess.run(
                    (loss, dice_per_class), feed_dict={x: batch['data'],y: batch['seg'], class_weights: cw})
                val_loss_running_mean += val_loss / cf.n_val_batches
                val_dices_running_batch_mean[0] += val_dices / cf.n_val_batches

            metrics['val']['loss'].append(val_loss_running_mean)
            metrics['val']['dices'] = np.append(metrics['val']['dices'], val_dices_running_batch_mean, axis=0)

            # perform tranining steps
            train_loss_running_mean = 0.
            train_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
            for _ in range(cf.n_train_batches):
                batch = next(batch_gen['train'])
                cw = utils.get_class_weights(batch['seg'])
                train_loss, train_dices, _ = sess.run(
                    (loss, dice_per_class, optimizer), feed_dict={x: batch['data'], y: batch['seg'], class_weights: cw})
                train_loss_running_mean += train_loss / cf.n_train_batches
                train_dices_running_batch_mean += train_dices / cf.n_train_batches

            metrics['train']['loss'].append(train_loss_running_mean)
            metrics['train']['dices'] = np.append(metrics['train']['dices'], train_dices_running_batch_mean, axis=0)

            #evaluate epoch
            val_loss = metrics['val']['loss'][-1]
            val_dices = metrics['val']['dices'][-1]
            fg_dice = np.mean(val_dices[1:])
            print "CHECK DICES", val_dices, fg_dice
            if val_loss < best_metrics['loss'][0]:
                best_metrics['loss'][0] = val_loss
                best_metrics['loss'][1] = epoch
            for cl in range(cf.n_classes):
                if val_dices[cl] > best_metrics['dices'][cl][0]:
                    best_metrics['dices'][cl][0] = val_dices[cl]
                    best_metrics['dices'][cl][1] = epoch
            if fg_dice > best_metrics['dices'][cf.n_classes][0]:
                best_metrics['dices'][cf.n_classes][0] = fg_dice
                best_metrics['dices'][cf.n_classes][1] = epoch
                saver.save(sess, os.path.join(cf.exp_dir, 'params_{}'.format(fold)))


            # monitor training
            TrainingPlot.update_and_save(metrics, best_metrics)
            batch = next(batch_gen['val'])
            soft_prediction = sess.run((predicter),
                                       feed_dict={x: batch['data'], is_training: False})
            correct_prediction = np.argmax(soft_prediction, axis=3)
            outfile = cf.plot_dir + '/pred_examle_{}.png'.format(fold)  ## FOLD!
            plot_batch_prediction(batch, correct_prediction, cf.n_classes, outfile)
            epoch += 1



def test():

    tf.reset_default_graph()

    # define tf graph
    x = tf.placeholder('float', shape=[None, cf.patch_size[0], cf.patch_size[0], cf.n_channels])
    logits, variables = create_UNet(x, features_root=cf.features_root, n_classes=cf.n_classes, is_training=False)
    predicter = tf.nn.softmax(logits)

    # initialize
    saver = tf.train.Saver()
    test_data_dict = data_loader.get_test_generator(cf)
    pred_dict = {key: [] for key in test_data_dict.keys()}

    with tf.Session() as sess:
        for fold in range(cf.n_cv_splits):
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(cf.exp_dir, 'params_{}'.format(fold)))

            for ix, pid in enumerate(test_data_dict.keys()):
                soft_prediction = sess.run(predicter, feed_dict={x: test_data_dict[pid]['data']})
                correct_prediction = np.argmax(soft_prediction, axis=3)
                dices =  utils.numpy_dice_per_class(utils.get_one_hot_prediction(correct_prediction, cf.n_classes), test_data_dict[pid]['seg'])
                pred_dict[pid].append(soft_prediction)
                print "DICES", dices, pid, fold
                plot_batch_prediction(test_data_dict[pid], correct_prediction, cf.n_classes,
                                      os.path.join(cf.plot_dir, '{}_pred_{}.png'.format(pid, fold)))

    print "FINALIZING"
    final_dices = []
    for ix, pid in enumerate(test_data_dict.keys()):
        print np.array(pred_dict[pid]).shape
        final_pred = np.argmax(np.mean(np.array(pred_dict[pid]),axis=0),axis=3)
        dices = utils.numpy_dice_per_class(utils.get_one_hot_prediction(final_pred, cf.n_classes), test_data_dict[pid]['seg'])
        print "FINAL DICES", dices, final_dices.append(dices)
        np.save(os.path.join(cf.exp_dir, '{}_pred_final.npy'.format(pid)), final_pred)
        plot_batch_prediction(test_data_dict[pid], final_pred, cf.n_classes,
                              os.path.join(cf.plot_dir, '{}_pred_final.png'.format(pid)))

    print "final dices", np.mean(final_dices, axis=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,  default='train') # ABREVIATIONS!
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--exp', type=str)
    mode = parser.parse_args().mode
    fold = parser.parse_args().fold
    exp_path = parser.parse_args().exp

    for dir in [cf.exp_dir, cf.test_dir, cf.plot_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)


    if mode=='train':
        shutil.copy(cf.__file__, os.path.join(cf.exp_dir, 'configs.py'))
        for fold in range(cf.n_cv_splits):
            train(fold)
    elif mode=='test':
        cf = imp.load_source('cf', os.path.join(exp_path, 'configs.py'))
        test()
    else:
        print "specified wrong execution mode..."