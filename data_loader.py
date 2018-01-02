__author__ = 'Paul F. Jaeger'

import numpy as np
import os
from sklearn.model_selection import KFold
from collections import OrderedDict
from batchgenerators.augmentations.utils import resize_image_by_padding, center_crop_2D_image, center_crop_3D_image
from batchgenerators.dataloading.data_loader import DataLoaderBase
from batchgenerators.transforms.spatial_transforms import Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import TransposeChannels, ConvertSegToOnehotTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform


def get_train_generators(cf, fold):

    train_val_data = load_dataset(cf)
    fg = get_cv_fold_ixs(len_data=len(train_val_data), seed=cf.seed)
    train_ix, val_ix = fg[fold]
    train_data = {k: v for ix, (k, v) in enumerate(train_val_data.iteritems()) if any(ix == s for s in train_ix)}
    val_data = {k: v for ix, (k, v) in enumerate(train_val_data.iteritems()) if any(ix == s for s in val_ix)}
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, do_aug=True)
    batch_gen['val'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)
    return batch_gen


def get_test_generator(cf):

    test_data = load_dataset(cf, split='test')
    test_data_dict = {}
    test_gen = create_data_gen_pipeline(test_data, cf=cf, test_pids=test_data.keys(), do_aug=False)
    for pid in test_data.keys():
        test_data_dict[pid] = next(test_gen)
    return test_data_dict


def load_dataset(cf, split='train', ids=()):

    in_dir = os.path.join(cf.root_dir, split)
    data_paths = [os.path.join(in_dir, f) for ix,f in enumerate(os.listdir(in_dir)) if (ix in ids) or len(ids)==0]
    concat_arr = [np.load(ii, mmap_mode='r') for ii in data_paths]
    pids = [ii.split('/')[-1].split('.')[0] for ii in data_paths]
    data = OrderedDict()
    for ix, pid in enumerate(pids):
        data[pid] = {'data': concat_arr[ix][..., 0], 'seg': concat_arr[ix][..., 1], 'pid': pid}
    return data


def get_cv_fold_ixs(len_data, seed):

    fold_list = []
    kf = KFold(n_splits=5, random_state=seed, shuffle=True,)
    for train_index, val_index in kf.split(range(len_data)):
        fold_list.append([train_index, val_index])
    return fold_list


def create_data_gen_pipeline(patient_data, cf, test_pids=None, do_aug=True):

    if test_pids is None:
        data_gen = BatchGenerator(patient_data, batch_size=cf.batch_size,
                                 pre_crop_size=cf.pre_crop_size, dim=cf.dim)
        cf.n_workers = 1
    else:
        data_gen = TestGenerator(patient_data, batch_size=cf.batch_size, n_batches=None,
                                 pre_crop_size=cf.pre_crop_size, test_pids=test_pids, dim=cf.dim)

    my_transforms = []
    if do_aug:
        mirror_transform = Mirror(axes=(2, 3))
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(ConvertSegToOnehotTransform(classes=(0, 1, 2)))
    my_transforms.append(TransposeChannels())
    all_transforms = Compose(my_transforms)
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator


class BatchGenerator(DataLoaderBase):

    def __init__(self, data, batch_size, pre_crop_size, n_batches=None, dim=2):
        super(BatchGenerator, self).__init__(data, batch_size,  n_batches)
        self.pre_crop_size = pre_crop_size
        self.dim = dim

    def generate_train_batch(self):

        patient_ix = np.random.choice(range(len(self._data)), self.BATCH_SIZE, replace=True)
        patient_ids = [self._data.keys()[ix] for ix in patient_ix]
        data, seg = [], []
        for b in range(self.BATCH_SIZE):

            patient = self._data[patient_ids[b]]
            shp = patient['data'].shape
            if self.dim == 2:
                slice_ix = np.random.choice(range(shp[2]))
                tmp_data = resize_image_by_padding(patient['data'][:, :, slice_ix], (
                    max(shp[0], self.pre_crop_size[0]), max(shp[1], self.pre_crop_size[1])), pad_value=0)
                tmp_seg = resize_image_by_padding(patient['seg'][:, :, slice_ix], (
                    max(shp[0], self.pre_crop_size[0]), max(shp[1], self.pre_crop_size[1])), pad_value=0)
                data.append(center_crop_2D_image(tmp_data, self.pre_crop_size)[np.newaxis])
                seg.append(center_crop_2D_image(tmp_seg, self.pre_crop_size)[np.newaxis])

            elif self.dim == 3:
                tmp_data = resize_image_by_padding(patient['data'], (
                    max(shp[0], self.pre_crop_size[0]), max(shp[1], self.pre_crop_size[1]),
                    max(shp[2], self.pre_crop_size[2])), pad_value=0)
                tmp_seg = resize_image_by_padding(patient['seg'], (
                    max(shp[0], self.pre_crop_size[0]), max(shp[1], self.pre_crop_size[1]),
                    max(shp[2], self.pre_crop_size[2])), pad_value=0)
                data.append(center_crop_3D_image(tmp_data, self.pre_crop_size)[np.newaxis])
                seg.append(center_crop_3D_image(tmp_seg, self.pre_crop_size)[np.newaxis])
        return {'data': np.array(data).astype('float32'), 'seg': np.array(seg).astype('float32'), 'pid': patient_ids}


class TestGenerator(DataLoaderBase):

    def __init__(self, data, batch_size, pre_crop_size, test_pids, n_batches=None, dim=2):
        super(TestGenerator, self).__init__(data, batch_size,  n_batches)
        self.pre_crop_size = pre_crop_size
        self.test_pids = test_pids
        self.dim = dim
        self.patient_ix = 0

    def generate_train_batch(self):

        pid = self.test_pids[self.patient_ix]
        patient = self._data[pid]
        shp = patient['data'].shape
        if self.dim == 2:
            z_pre_crop_size = shp[2]
        else:
            z_pre_crop_size = max(shp[2], self.pre_crop_size[2])

        data_arr = resize_image_by_padding(patient['data'], (
            max(shp[0], self.pre_crop_size[0]), max(shp[1], self.pre_crop_size[1]), z_pre_crop_size), pad_value=0)
        seg_arr = resize_image_by_padding(patient['seg'], (
            max(shp[0], self.pre_crop_size[0]), max(shp[1], self.pre_crop_size[1]), z_pre_crop_size), pad_value=0)
        data_arr = center_crop_3D_image(data_arr, (self.pre_crop_size[0], self.pre_crop_size[0], z_pre_crop_size))[np.newaxis]
        seg_arr = center_crop_3D_image(seg_arr, (self.pre_crop_size[0], self.pre_crop_size[0], z_pre_crop_size))[np.newaxis]

        if self.dim == 2:
            data_arr = np.transpose(data_arr, axes=(3, 0, 1, 2))
            seg_arr = np.transpose(seg_arr, axes=(3, 0, 1, 2))
        else:
            data_arr = data_arr[np.newaxis]
            seg_arr = seg_arr[np.newaxis]

        self.patient_ix += 1
        if self.patient_ix == len(self.test_pids):
            raise StopIteration
        return {'data': data_arr.astype('float32'), 'seg': seg_arr.astype('float32'), 'pid': pid}


