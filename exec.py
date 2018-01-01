__author__ = 'Paul F. Jaeger'

import argparse
import imp
import os
import time

import numpy as np
import tensorflow as tf

import model
import utils
import data_loader
from plotting import TrainingPlot_2Panel, plot_batch_prediction


#DISCLAIMER


def train(fold):

    utils.prep_exp(cf)
    logger = utils.get_logger(cf)
    logger.info('intitializing tensorflow graph...')
    tf.reset_default_graph()
    x = tf.placeholder('float', shape=cf.network_input_shape)
    y = tf.placeholder('float', shape=cf.network_output_shape)
    learning_rate = tf.Variable(cf.learning_rate) #should this stay a variable??? if im not using it?
    logits, variables = model.create_UNet(x, features_root=cf.n_features_root, n_classes=cf.n_classes, dim=cf.dim)
    if cf.loss_name == 'weighted_cross_entropy':
        class_weights = tf.placeholder('float') # has to work a different way. try later!
        loss = utils._get_loss(logits, y, cf.n_classes, cf.loss_name, class_weights, cf.dim)
    else:
        loss = utils._get_loss(logits, y, cf.n_classes, cf.loss_name, cf.dim)
    predicter = tf.nn.softmax(logits)
    dice_per_class = utils.get_dice_per_class(logits, y, dim=cf.dim)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    saver = tf.train.Saver()

    # set up training
    metrics = {}
    metrics['train'] = {'loss': [0.], 'dices': np.zeros(shape=(1, cf.n_classes))}
    metrics['val'] = {'loss': [0.], 'dices': np.zeros(shape=(1, cf.n_classes))}
    best_metrics = {'loss': [10, 0], 'dices': np.zeros(shape=(cf.n_classes + 1, 2))}
    file_name = cf.exp_dir + '/monitor_{}.png'.format(fold)
    TrainingPlot = TrainingPlot_2Panel(cf.n_epochs, file_name, cf.experiment_name, cf.class_dict)

    logger.info('initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, fold)

    logger.info('starting training...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(cf.n_epochs):

            start_time = time.time()
            # perform validation
            val_loss_running_mean = 0.
            val_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
            for _ in range(cf.n_val_batches):
                batch = next(batch_gen['val'])
                if cf.class_weights:
                    cw = utils.get_class_weights(batch['seg'])
                    feed_dict = {x: batch['data'], y: batch['seg'], class_weights: cw}
                else:
                    feed_dict = {x: batch['data'], y: batch['seg']}
                val_loss, val_dices = sess.run(
                    (loss, dice_per_class), feed_dict=feed_dict)
                val_loss_running_mean += val_loss / cf.n_val_batches
                val_dices_running_batch_mean[0] += val_dices / cf.n_val_batches
            metrics['val']['loss'].append(val_loss_running_mean)
            metrics['val']['dices'] = np.append(metrics['val']['dices'], val_dices_running_batch_mean, axis=0)

            # perform tranining steps
            train_loss_running_mean = 0.
            train_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
            for _ in range(cf.n_train_batches):
                batch = next(batch_gen['train'])
                if cf.class_weights:
                    cw = utils.get_class_weights(batch['seg'])
                    print "WEIGHTS", cw.shape, cw
                    feed_dict = {x: batch['data'], y: batch['seg'], class_weights: cw}
                else:
                    feed_dict = {x: batch['data'], y: batch['seg']}
                train_loss, train_dices, _ = sess.run(
                    (loss, dice_per_class, optimizer), feed_dict=feed_dict)
                train_loss_running_mean += train_loss / cf.n_train_batches
                train_dices_running_batch_mean += train_dices / cf.n_train_batches
            metrics['train']['loss'].append(train_loss_running_mean)
            metrics['train']['dices'] = np.append(metrics['train']['dices'], train_dices_running_batch_mean, axis=0)

            #evaluate epoch
            val_loss = metrics['val']['loss'][-1]
            val_dices = metrics['val']['dices'][-1]
            fg_dice = np.mean(val_dices[1:])
            logger.info('trained epoch {e}: val_loss {l}, val_dices: {d}'.format(e=epoch, l=val_loss, d=val_dices))
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
            soft_prediction = sess.run((predicter), feed_dict={x: batch['data']})
            correct_prediction = np.argmax(soft_prediction, axis=-1)
            outfile = cf.plot_dir + '/pred_example_{}.png'.format(fold)
            plot_batch_prediction(batch, correct_prediction, cf.n_classes, outfile, dim=cf.dim)
            logger.info('trained epoch {e}: val_loss {l}, val_dices: {d}, took {t} sec.'.format(
                e=epoch, l=np.round(val_loss, 3), d=val_dices, t=np.round(time.time() - start_time, 0)))
            epoch += 1



def test(folds, ):

    logger = utils.get_logger(cf)
    logger.info('performing testing in {d}D over fold(s) {f} on experiment {e}'.format(d=cf.dim, f=folds, e=cf.exp_dir))
    logger.info('intitializing tensorflow graph...')
    tf.reset_default_graph()
    x = tf.placeholder('float', shape=cf.network_input_shape)
    logits, variables = model.create_UNet(x, features_root=cf.n_features_root, n_classes=cf.n_classes, dim=cf.dim)
    predicter = tf.nn.softmax(logits)
    saver = tf.train.Saver()


    logger.info('intitializing test generator...')
    test_data_dict = data_loader.get_test_generator(cf)
    pred_dict = {key: [] for key in test_data_dict.keys()}

    logger.info('starting testing...')
    with tf.Session() as sess:
        for fold in folds:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(cf.exp_dir, 'params_{}'.format(fold)))

            for ix, pid in enumerate(test_data_dict.keys()):
                for m in range(4):
                    if m == 0:
                        test_arr = test_data_dict[pid]['data']
                        print "CHECK SHAPE", test_arr.shape
                        soft_prediction = sess.run(predicter, feed_dict={x: test_arr})
                    if m == 1:
                        test_arr = np.flip(test_data_dict[pid]['data'], axis=cf.dim-1)
                        soft_prediction = np.flip(sess.run(predicter, feed_dict={x: test_arr}),axis=cf.dim-1)
                    if m == 2:
                        test_arr = np.flip(test_data_dict[pid]['data'], axis=cf.dim)
                        soft_prediction = np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=cf.dim)
                    if m == 3:
                        test_arr = np.flip(np.flip(test_data_dict[pid]['data'], axis=cf.dim-1), axis=cf.dim)
                        soft_prediction = np.flip(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=cf.dim-1), axis=cf.dim)
                    seg_arr = test_data_dict[pid]['seg']
                    correct_prediction = np.argmax(soft_prediction, axis=-1)
                    dices =  utils.numpy_volume_dice_per_class(utils.get_one_hot_prediction(correct_prediction, cf.n_classes), seg_arr)
                    pred_dict[pid].append(soft_prediction)
                    logger.info('starting testing...{} {} {} {}'.format(dices, pid, fold , m))
                    # plot_batch_prediction(test_data_dict[pid], correct_prediction, cf.n_classes, #kommt dann weg
                    #                       os.path.join(cf.plot_dir, 'pred_{}.png'.format(fold)), dim=cf.dim)

        logger.info('finalizing...')
    final_dices = []
    for ix, pid in enumerate(test_data_dict.keys()):
        final_pred_soft = np.mean(np.array(pred_dict[pid]),axis=0)
        final_pred_correct = np.argmax(final_pred_soft, axis=-1)
        avg_dices = utils.numpy_volume_dice_per_class(utils.get_one_hot_prediction(final_pred_correct, cf.n_classes), test_data_dict[pid]['seg'])
        final_dices.append(avg_dices)
        logger.info('avg dices... {} over {} preds'.format(avg_dices, len(pred_dict[pid])))
        np.save(os.path.join(cf.exp_dir, '{}_pred_final.npy'.format(pid)), final_pred_soft)
        plot_batch_prediction(test_data_dict[pid], final_pred_correct, cf.n_classes,
                              os.path.join(cf.plot_dir, '{}_pred_final.png'.format(pid)), dim=cf.dim)

    logger.info('final dices {}'.format(np.mean(final_dices, axis=0)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,  default='train') # ABREVIATIONS!
    parser.add_argument('--folds', nargs='+', type=int, default=[0])
    parser.add_argument('--exp', type=str)
    mode = parser.parse_args().mode
    folds = parser.parse_args().folds
    exp_path = parser.parse_args().exp

    if mode=='train':
        cf = imp.load_source('cf', 'configs.py')
        for fold in folds:
            train(fold)
    elif mode=='test':
        cf = imp.load_source('cf', os.path.join(exp_path, 'configs.py')) #merge configs with if dim for sure.
        test(folds)
    else:
        print 'specified wrong execution mode in args...'