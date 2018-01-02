__author__ = 'Paul F. Jaeger'


# DO NOT SUBMIT THIS FILE!!!!



import os
import numpy as np
import imp
import utils
import data_loader


def get_patient_dices_ensemble(path_2D, path_3D):



    cf = imp.load_source('cf', os.path.join(path_2D, 'configs.py'))
    test_data_dict = data_loader.get_test_generator(cf)
    final_dices = []

    for ix, pid in enumerate(test_data_dict.keys()):

        in_2D = np.load(os.path.join(path_2D, pid + '_pred_final.npy'), mmap_mode='r')
        in_3D = np.load(os.path.join(path_3D, pid + '_pred_final.npy'), mmap_mode='r')
        pred_2D = in_2D[0]
        pred_3D = in_3D[0, 0]
        seg = in_2D[1]
        # print "2D", pred_2D.shape
        # print "3D", pred_3D.shape
        # print "seg", seg.shape

        final_pred = np.argmax(np.mean(np.array([pred_2D, pred_3D]), axis=0), axis=-1)
        # print "FINAL PRED", final_pred.shape
        avg_dices = utils.numpy_volume_dice_per_class(utils.get_one_hot_prediction(final_pred, cf.n_classes),
                                                      seg) #how is this in dims? is ok.

        final_dices.append(avg_dices)
        # print 'avg dices... {} over {} preds'.format(avg_dices, pid)
        # np.save(os.path.join(cf.exp_dir, '{}_pred_final.npy'.format(pid)), final_pred)
        # plot_batch_prediction(test_data_dict[pid], final_pred, cf.n_classes,
        #                       os.path.join(cf.plot_dir, '{}_pred_final.png'.format(pid)), dim=cf.dim)

    return np.array(final_dices)


def get_patient_dices(path, dim=2):


    cf = imp.load_source('cf', os.path.join(path, 'configs.py'))
    test_data_dict = data_loader.get_test_generator(cf)
    final_dices = []
    for ix, pid in enumerate(test_data_dict.keys()):
        in_arr = np.load(os.path.join(path, pid + '_pred_final.npy'), mmap_mode='r')
        pred = in_arr[0]
        seg = in_arr[1]
        if dim==3:
            pred = pred[0]
            seg = seg[0]
        # print "pred", pred.shape
        # print "seg", seg.shape

        final_pred = np.argmax(pred, axis=-1)
        avg_dices = utils.numpy_volume_dice_per_class(utils.get_one_hot_prediction(final_pred, cf.n_classes),seg)
        final_dices.append(avg_dices)
        # print 'avg dices... {} over {} preds'.format(avg_dices, pid)
    return np.array(final_dices)





if __name__ == '__main__':

    from scipy.stats import wilcoxon
    from scipy.stats import ttest_rel

    path_3D = '/mnt/hdd/experiments/segmentation/final_dice_3D_good'
    path_batched = '/mnt/hdd/experiments/segmentation/final_2D_dice_lr3_32f_10bs'
    path_sliced = '/mnt/hdd/experiments/segmentation/last_check_dice_sliced_2D'

    print "TEST RESULTS"

    ens = get_patient_dices_ensemble(path_batched, path_3D)
    print 'ENS {} {}'.format(np.mean(ens, axis=0), np.std(ens, axis=0))
    ens = ens/(2-ens)
    print 'ENS {} {}'.format(np.mean(ens, axis=0), np.std(ens, axis=0))

    sliced = get_patient_dices(path_sliced)
    print 'SLICED {} {}'.format(np.mean(sliced, axis=0), np.std(sliced, axis=0))
    sliced = sliced / (2 - sliced)
    print 'SLICED {} {}'.format(np.mean(sliced, axis=0), np.std(sliced, axis=0))

    batched = get_patient_dices(path_batched)
    print 'BATCHED  {} {}'.format(np.mean(batched, axis=0), np.std(batched, axis=0))
    batched = batched / (2 - batched)
    print 'BATCHED  {} {}'.format(np.mean(batched, axis=0), np.std(batched, axis=0))

    threeD = get_patient_dices(path_3D, dim=3)
    print '3D  {} {}'.format(np.mean(threeD, axis=0), np.std(threeD, axis=0))
    threeD = threeD / (2 - threeD)
    print '3D {} {}'.format(np.mean(threeD, axis=0), np.std(threeD, axis=0))

    print "SLICE VS BATCH"
    print wilcoxon(sliced[:, 0], batched[:, 0])
    print wilcoxon(sliced[:, 1], batched[:, 1])
    print wilcoxon(sliced[:, 2], batched[:, 2])
    print
    print ttest_rel(sliced[:, 0], batched[:, 0])
    print ttest_rel(sliced[:, 1], batched[:, 1])
    print ttest_rel(sliced[:, 2], batched[:, 2])

    print
    print "CV RESULTS"
    threeD = np.load(os.path.join(path_3D, 'test', 'val_dices.npy'), mmap_mode='r')
    sliced = np.load(os.path.join(path_sliced, 'test', 'val_dices_natural.npy'), mmap_mode='r')
    batched = np.load(os.path.join(path_batched, 'test', 'val_dices_natural.npy'), mmap_mode='r')

    print 'SLICED {} {}'.format(np.mean(sliced, axis=0), np.std(sliced, axis=0))
    sliced = sliced / (2 - sliced)
    print 'SLICED {} {}'.format(np.mean(sliced, axis=0), np.std(sliced, axis=0))
    print sliced.shape

    print 'BATCHED {} {}'.format(np.mean(batched, axis=0), np.std(batched, axis=0))
    batched = batched / (2 - batched)
    print 'BATCHED {} {}'.format(np.mean(batched, axis=0), np.std(batched, axis=0))
    print batched.shape

    threeD = get_patient_dices(path_3D, dim=3)
    print '3D {} {}'.format(np.mean(threeD, axis=0), np.std(threeD, axis=0))
    threeD = threeD / (2 - threeD)
    print '3D {} {}'.format(np.mean(threeD, axis=0), np.std(threeD, axis=0))


    print "SLICE VS BATCH"
    print wilcoxon(sliced[:, 0], batched[:, 0])
    print wilcoxon(sliced[:, 1], batched[:, 1])
    print wilcoxon(sliced[:, 2], batched[:, 2])
    print
    print ttest_rel(sliced[:, 0], batched[:, 0])
    print ttest_rel(sliced[:, 1], batched[:, 1])
    print ttest_rel(sliced[:, 2], batched[:, 2])