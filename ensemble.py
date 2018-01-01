__author__ = 'Paul F. Jaeger'


# DO NOT SUBMIT THIS FILE!!!!

path_3D = '/mnt/hdd/experiments/segmentation/final_dice_3D_good'
path_2D = '/mnt/hdd/experiments/segmentation/final_2D_dice_lr3_32f_10bs'

import os

import numpy as np

import configs as cf
import utils
from dm.nci_prostate import data_loader

test_data_dict = data_loader.get_test_generator(cf)
final_dices = []


for ix, pid in enumerate(test_data_dict.keys()):

    pred_2D = np.load(os.path.join(path_2D, pid + '_pred_final.npy'), mmap_mode='r')
    pred_3D = np.load(os.path.join(path_3D, pid + '_pred_final.npy'), mmap_mode='r')[0]
    seg = test_data_dict[pid]['seg']
    print "2D", pred_2D.shape
    print "3D", pred_3D.shape
    print "seg", seg.shape

    final_pred = np.argmax(np.mean(np.array([pred_2D, pred_3D]), axis=0), axis=-1)
    print "FINAL PRED", final_pred.shape
    avg_dices = utils.numpy_volume_dice_per_class(utils.get_one_hot_prediction(final_pred, cf.n_classes),
                                                  seg) #how is this in dims? is ok.

    final_dices.append(avg_dices)
    print 'avg dices... {} over {} preds'.format(avg_dices, pid)

print 'final dices {}'.format(np.mean(final_dices, axis=0))
    # np.save(os.path.join(cf.exp_dir, '{}_pred_final.npy'.format(pid)), final_pred)
    # plot_batch_prediction(test_data_dict[pid], final_pred, cf.n_classes,
    #                       os.path.join(cf.plot_dir, '{}_pred_final.png'.format(pid)), dim=cf.dim)