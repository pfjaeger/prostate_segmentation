__author__ = 'Paul F. Jaeger'

import os
import dicom
import dicom_numpy
import nrrd
import numpy as np
from skimage.transform import resize


def collectPaths(root_dir, out_dir, split):

    split_dir = os.path.join(root_dir, split)
    split_dir_seg = split_dir + '-segm'
    out_split_dir = os.path.join(out_dir, split) if split is not 'leaderboard' else os.path.join(out_dir, 'train')
    if not os.path.exists(out_split_dir):
        os.mkdir(out_split_dir)

    seg_paths = [os.path.join(split_dir_seg, ii) for ii in os.listdir(split_dir_seg)]
    ix = 0
    for path, dirs, files in os.walk(split_dir):
        if len(files) > 0:

            dicom_slices = [dicom.read_file(os.path.join(path, f)) for f in files]
            img_arr, img_affine = dicom_numpy.combine_slices(dicom_slices)

            pid = path.split('/')[-3]
            seg_path = [seg_path for seg_path in seg_paths if pid in seg_path][0]
            seg_arr, seg_info = nrrd.read(seg_path)
            if img_arr.shape==seg_arr.shape:
                data_arr = np.concatenate((img_arr[:, :, :, np.newaxis], seg_arr[:, :, :, np.newaxis]), axis=3)
                rs_data_arr = resample_array(data_arr, img_affine)
                np.save(os.path.join(out_split_dir, '{}.npy'.format(path.split('/')[-3])), rs_data_arr)
                print "processed", ix, os.path.join(out_split_dir, '{}.npy'.format(path.split('/')[-3]))
            else:
                print "failed to process due to shape mismatch:", img_arr.shape, seg_arr.shape, pid, split
            ix += 1


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


    root_dir = '/mnt/hdd/data/dm/'
    out_dir = os.path.join(root_dir, 'numpy_arrays')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    collectPaths(root_dir, out_dir, 'train')
    # collectPaths(root_dir, out_dir, 'leaderboard')
    # collectPaths(root_dir, out_dir, 'test')