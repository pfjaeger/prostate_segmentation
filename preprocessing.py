__author__ = 'Paul F. Jaeger'

import configs as cf
import os
import dicom
import dicom_numpy
import nrrd
import numpy as np
from skimage.transform import resize


def preprocess_dataset(raw_data_dir, out_dir, set):
    """
    load and concat the raw dicom/nrrd data, resample it to equalize the spacings and save out
    the numpy arrays per patient. Additionally, determine the class_weights as 1-class_ratio
    """

    set_dir = os.path.join(raw_data_dir, set)
    set_dir_seg = set_dir + '-segm'
    out_set_dir = os.path.join(out_dir, set) if set is not 'leaderboard' else os.path.join(out_dir, 'train')
    if not os.path.exists(out_set_dir):
        os.mkdir(out_set_dir)

    seg_paths = [os.path.join(set_dir_seg, ii) for ii in os.listdir(set_dir_seg)]
    collect_class_weights = []
    ix = 0
    for path, dirs, files in os.walk(set_dir):
        if len(files) > 0:
            dicom_slices = [dicom.read_file(os.path.join(path, f)) for f in files]
            img_arr, img_affine = dicom_numpy.combine_slices(dicom_slices)
            pid = path.split('/')[-3]
            seg_path = [seg_path for seg_path in seg_paths if pid in seg_path][0]
            seg_arr, seg_info = nrrd.read(seg_path)
            if img_arr.shape == seg_arr.shape:
                data_arr = np.concatenate((img_arr[:, :, :, np.newaxis], seg_arr[:, :, :, np.newaxis]), axis=3)
                rs_data_arr = resample_array(data_arr, img_affine)
                class_ratios = np.unique(rs_data_arr[..., 1], return_counts=True)[1]/float(rs_data_arr[...,1].size)
                collect_class_weights.append(class_ratios)
                np.save(os.path.join(out_set_dir, '{}.npy'.format(path.split('/')[-3])), rs_data_arr)
                print "processed", ix, os.path.join(out_set_dir, '{}.npy'.format(path.split('/')[-3]))
            else:
                print "failed to process due to shape mismatch: {} {} {} {}".format(
                    img_arr.shape, seg_arr.shape, pid, set)
            ix += 1
    class_weights = 1 - np.mean(np.array(collect_class_weights), axis=0)
    print "class weights for set {} and classes BG, PZ, GC:".format(set), class_weights


def resample_array(src_img, img_affine, target_spacing=0.4):

    target_shape_x = int(src_img.shape[0] * img_affine[0][0] / target_spacing)
    target_shape_y = int(src_img.shape[1] * img_affine[1][1] / target_spacing)
    out_array = np.zeros((target_shape_x, target_shape_y, src_img.shape[2], src_img.shape[3]))

    for slc in range(out_array.shape[2]):
        for mod in range(out_array.shape[3]):
            out_array[:, :, slc, mod] = np.round(
            resize(src_img[:, :, slc, mod].astype(float), out_array.shape[:2], order=3, preserve_range=True,
                   mode='edge')).astype('float32')
    return out_array


if __name__ == "__main__":

    if not os.path.exists(cf.pp_data_dir):
        os.mkdir(cf.pp_data_dir)

    data_sets = ['train', 'leaderboard', 'test']
    for ds in data_sets:
        preprocess_dataset(cf.raw_data_dir, cf.pp_data_dir, ds)
