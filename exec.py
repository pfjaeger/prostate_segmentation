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
    """
    performs the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """

    # set up experiment dirs and copy config file
    utils.prep_exp(cf)

    logger = utils.get_logger(cf)
    logger.info('performing training in {d}D over fold {f} on experiment {e}'.format(d=cf.dim, f=fold, e=cf.exp_dir))
    logger.info('intitializing tensorflow graph...')

    tf.reset_default_graph()
    x = tf.placeholder('float', shape=cf.network_input_shape)
    y = tf.placeholder('float', shape=cf.network_output_shape)
    learning_rate = tf.Variable(cf.learning_rate)
    logits = model.create_UNet(x, cf.n_features_root, cf.n_classes, dim=cf.dim, logger=logger)
    if cf.loss_name == 'weighted_cross_entropy':
        class_weights = tf.placeholder('float')
        loss = utils._get_loss(logits, y, cf.n_classes, cf.loss_name, class_weights, cf.dim)
    else:
        loss = utils._get_loss(logits, y, cf.n_classes, cf.loss_name, cf.dim)
    predicter = tf.nn.softmax(logits)
    dice_per_class = utils.get_dice_per_class(logits, y, dim=cf.dim)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    saver = tf.train.Saver()

    # set up training
    metrics = {}
    metrics['train'] = {'loss': [None], 'dices': np.zeros(shape=(1, cf.n_classes))} # CHECK IF THIS WORKS
    metrics['val'] = {'loss': [None], 'dices': np.zeros(shape=(1, cf.n_classes))}
    best_metrics = {'loss': [10, 0], 'dices': np.zeros(shape=(cf.n_classes + 1, 2))}
    file_name = cf.plot_dir + '/monitor_{}.png'.format(fold)
    TrainingPlot = TrainingPlot_2Panel(cf.n_epochs, file_name, cf.experiment_name, cf.class_dict)

    logger.info('initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, fold)

    logger.info('starting training...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(cf.n_epochs):

            start_time = time.time()

            # perform tranining steps
            train_loss_running_mean = 0.
            train_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
            for _ in range(cf.n_train_batches):
                batch = next(batch_gen['train'])
                if cf.loss_name == 'weighted_cross_entropy':
                    cw = utils.get_class_weights(batch['seg'])
                    feed_dict = {x: batch['data'], y: batch['seg'], class_weights: cw}
                else:
                    feed_dict = {x: batch['data'], y: batch['seg']}
                train_loss, train_dices, _ = sess.run(
                    (loss, dice_per_class, optimizer), feed_dict=feed_dict)
                train_loss_running_mean += train_loss / cf.n_train_batches
                train_dices_running_batch_mean += train_dices / cf.n_train_batches
            metrics['train']['loss'].append(train_loss_running_mean)
            metrics['train']['dices'] = np.append(metrics['train']['dices'], train_dices_running_batch_mean, axis=0)

            # perform validation
            val_loss_running_mean = 0.
            val_dices_running_batch_mean = np.zeros(shape=(1, cf.n_classes))
            for _ in range(cf.n_val_batches):
                batch = next(batch_gen['val'])
                if cf.loss_name == 'weighted_cross_entropy':
                    cw = utils.get_class_weights(batch['seg']) # DO GLOBALLY
                    feed_dict = {x: batch['data'], y: batch['seg'], class_weights: cw}
                else:
                    feed_dict = {x: batch['data'], y: batch['seg']}
                val_loss, val_dices = sess.run(
                    (loss, dice_per_class), feed_dict=feed_dict)
                val_loss_running_mean += val_loss / cf.n_val_batches
                val_dices_running_batch_mean[0] += val_dices / cf.n_val_batches
            metrics['val']['loss'].append(val_loss_running_mean)
            metrics['val']['dices'] = np.append(metrics['val']['dices'], val_dices_running_batch_mean, axis=0)

            #evaluate epoch
            val_loss = metrics['val']['loss'][-1]
            val_dices = metrics['val']['dices'][-1]
            if val_loss < best_metrics['loss'][0]:
                best_metrics['loss'][0] = val_loss
                best_metrics['loss'][1] = epoch
            for cl in range(cf.n_classes):
                if val_dices[cl] > best_metrics['dices'][cl][0]:
                    best_metrics['dices'][cl][0] = val_dices[cl]
                    best_metrics['dices'][cl][1] = epoch

            # selection criterion is the averaged dice over both foreground classes
            fg_dice = np.mean(val_dices[1:])
            if fg_dice > best_metrics['dices'][cf.n_classes][0]:
                best_metrics['dices'][cf.n_classes][0] = fg_dice
                best_metrics['dices'][cf.n_classes][1] = epoch
                saver.save(sess, os.path.join(cf.exp_dir, 'params_{}'.format(fold)))

            # plot updated monitoring and prediction example
            TrainingPlot.update_and_save(metrics, best_metrics)
            batch = next(batch_gen['val'])
            correct_prediction = np.argmax(sess.run((predicter), feed_dict={x: batch['data']}), axis=-1)
            outfile = cf.plot_dir + '/pred_example_{}.png'.format(fold) #set fold -> epoch to keep plots from all epochs
            plot_batch_prediction(batch, correct_prediction, cf.n_classes, outfile, dim=cf.dim)
            logger.info('trained epoch {e}: val_loss {l}, val_dices: {d}, took {t} sec.'.format(
                e=epoch, l=np.round(val_loss, 3), d=val_dices, t=np.round(time.time() - start_time, 0)))


def test(folds):
    """
    performs
    """
    logger = utils.get_logger(cf)
    logger.info('performing testing in {d}D over fold(s) {f} on experiment {e}'.format(d=cf.dim, f=folds, e=cf.exp_dir))
    logger.info('intitializing tensorflow graph...')
    tf.reset_default_graph()
    x = tf.placeholder('float', shape=cf.network_input_shape)
    logits, variables = model.create_UNet(x, cf.n_features_root, cf.n_classes, dim=cf.dim, logger=logger)
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
                patient_fold_prediction = []
                test_arr = test_data_dict[pid]['data']
                patient_fold_prediction.append(sess.run(predicter, feed_dict={x: test_arr}))

                test_arr = np.flip(test_data_dict[pid]['data'], axis=cf.dim-1)
                patient_fold_prediction.append(np.flip(sess.run(predicter, feed_dict={x: test_arr}),axis=cf.dim-1))

                test_arr = np.flip(test_data_dict[pid]['data'], axis=cf.dim)
                patient_fold_prediction.append(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=cf.dim))

                test_arr = np.flip(np.flip(test_data_dict[pid]['data'], axis=cf.dim-1), axis=cf.dim)
                patient_fold_prediction.append(np.flip(np.flip(sess.run(predicter, feed_dict={x: test_arr}), axis=cf.dim-1), axis=cf.dim))

                pred_dict[pid].append(np.mean(np.array(patient_fold_prediction), axis=0))

    logger.info('finalizing predictions...')
    final_dices = []
    for ix, pid in enumerate(test_data_dict.keys()):
        final_pred_soft = np.mean(np.array(pred_dict[pid]), axis=0)
        final_pred_correct = np.argmax(final_pred_soft, axis=-1)
        seg = test_data_dict[pid]['seg']
        avg_dices = utils.numpy_volume_dice_per_class(utils.get_one_hot_prediction(final_pred_correct, cf.n_classes), seg)
        final_dices.append(avg_dices)
        logger.info('avg dices for patient {p} over {a} preds: {d}'.format(p=pid, a=len(pred_dict[pid]), d=avg_dices))
        # np.save(os.path.join(cf.test_dir, '{}_pred_final.npy'.format(pid)), np.concatenate((final_pred_soft[np.newaxis], seg[np.newaxis])))
        # plot_batch_prediction(test_data_dict[pid], final_pred_correct, cf.n_classes,
        #                       os.path.join(cf.test_dir, '{}_pred_final.png'.format(pid)), dim=cf.dim)

    logger.info('final dices mean: {}'.format(np.mean(final_dices, axis=0)))
    logger.info('final dices std: {}'.format(np.std(final_dices, axis=0)))


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,  default='train', help='must be set to train or test mode.')
    parser.add_argument('--folds', nargs='+', type=int, default=[0], help='sets either folds to train or folds to test on.')
    parser.add_argument('--exp', type=str, help='sets the directory path of the experiment to be tested.')
    mode = parser.parse_args().mode
    folds = parser.parse_args().folds
    exp_path = parser.parse_args().exp

    if mode == 'train':
        cf = imp.load_source('cf', 'configs.py')
        for fold in folds:
            train(fold)
    elif mode == 'test':
        cf = imp.load_source('cf', os.path.join(exp_path, 'configs.py'))
        test(folds)
    else:
        raise ValueError('specified wrong execution mode in parsed args...')