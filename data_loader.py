__author__ = 'Paul F. Jaeger'

import numpy as np
import os
from batchgenerators.augmentations.utils import resize_image_by_padding, center_crop_2D_image, center_crop_3D_image
from batchgenerators.dataloading.data_loader import DataLoaderBase
from batchgenerators.transforms.spatial_transforms import Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import TransposeChannels, ConvertSegToOnehotTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from sklearn.model_selection import KFold
from collections import OrderedDict




def get_train_generators(cf, fold):

    all_data = load_NCI_ISBI_dataset(cf)
    fg = get_cv_fold_ixs(len_data=len(all_data), seed=cf.seed)
    train_ix, val_ix = fg[fold]
    train_data = {k:v for ix, (k, v) in enumerate(all_data.iteritems()) if any(ix==s for s in train_ix)}
    val_data = {k:v for ix, (k, v) in enumerate(all_data.iteritems()) if any(ix==s for s in val_ix)}
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, do_aug=True)
    batch_gen['val'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)
    print "CHECK FOLD", len(all_data), len(train_data), len(val_data), val_ix
    return batch_gen


def get_test_generator(cf):

    test_data = load_NCI_ISBI_dataset(cf, split='test')
    test_data_dict = {}
    for ix, pid in enumerate(test_data.keys()):
        test_gen = create_data_gen_pipeline(test_data, cf=cf, test_pid=pid, do_aug=False)
        test_data_dict[pid] = next(test_gen)
    return test_data_dict



def load_NCI_ISBI_dataset(cf, split='train', ids=()):

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


def create_data_gen_pipeline(patient_data, cf, test_pid=None, do_aug=True):


    if test_pid is None:
        data_gen = BatchGenerator(patient_data, BATCH_SIZE=cf.batch_size,
                                 PATCH_SIZE=cf.pad_size, dim=cf.dim)

    else:
        data_gen = TestGenerator(patient_data, BATCH_SIZE=cf.batch_size, n_batches=None,
                                 PATCH_SIZE=cf.pad_size, test_pid=test_pid, dim=cf.dim)



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

    def __init__(self, data, BATCH_SIZE, PATCH_SIZE, n_batches=None, dim=2):
        super(BatchGenerator, self).__init__(data, BATCH_SIZE,  n_batches)
        self.PATCH_SIZE = PATCH_SIZE
        self.dim = dim

    def generate_train_batch(self):

        patient_ix = np.random.choice(range(len(self._data)), self.BATCH_SIZE, replace=True)
        patient_ids = [self._data.keys()[ix] for ix in patient_ix]
        data, seg = [], []
        for b in range(self.BATCH_SIZE):

            if self.dim == 2:
                shp = self._data[patient_ids[b]]['data'].shape
                slice_ix = np.random.choice(range(shp[2]))
                tmp_data = resize_image_by_padding(self._data[patient_ids[b]]['data'][:, :, slice_ix], (
                max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1])), pad_value=0)
                tmp_seg = resize_image_by_padding(self._data[patient_ids[b]]['seg'][:, :, slice_ix],
                                                  (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1])),
                                                  pad_value=0)
                data.append(center_crop_2D_image(tmp_data, self.PATCH_SIZE)[np.newaxis])
                seg.append(center_crop_2D_image(tmp_seg, self.PATCH_SIZE)[np.newaxis])

            elif self.dim == 3:
                shp = self._data[patient_ids[b]]['data'].shape
                tmp_data = resize_image_by_padding(self._data[patient_ids[b]]['data'], (
                    max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), max(shp[2], self.PATCH_SIZE[2])),
                                                   pad_value=0)
                tmp_seg = resize_image_by_padding(self._data[patient_ids[b]]['seg'],
                                                  (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]),
                                                   max(shp[2], self.PATCH_SIZE[2])),
                                                  pad_value=0)
                data.append(center_crop_3D_image(tmp_data, self.PATCH_SIZE)[np.newaxis])
                seg.append(center_crop_3D_image(tmp_seg, self.PATCH_SIZE)[np.newaxis])

        return {'data': np.array(data).astype('float32'), 'seg': np.array(seg).astype('float32'), 'pid': patient_ids}



class TestGenerator(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE, PATCH_SIZE, test_pid, n_batches=None, dim=2):
        super(TestGenerator, self).__init__(data, BATCH_SIZE,  n_batches)
        self.PATCH_SIZE = PATCH_SIZE
        self.test_pid = test_pid
        self.dim = dim

    def generate_train_batch(self): #naming!?

            shp = self._data[self.test_pid]['data'].shape
            #actually i dont need the padding since all imgs are bigger than patch size.
            #no padding cropping in z. becuase testing! do this in 3D sincein train and test 1 batch is 1 patient volume.
            if self.dim==2:
                z_pad_size = shp[2]#max(shp[2], 32)
            else:
                z_pad_size = max(shp[2], self.PATCH_SIZE[2])

            data_arr = resize_image_by_padding(self._data[self.test_pid]['data'],
                                               (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), z_pad_size), pad_value=0)
            seg_arr = resize_image_by_padding(self._data[self.test_pid]['seg'],
                                               (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), z_pad_size), pad_value=0)
            data_arr  = center_crop_3D_image(data_arr, (self.PATCH_SIZE[0], self.PATCH_SIZE[0], z_pad_size))[np.newaxis]
            seg_arr = center_crop_3D_image(seg_arr, (self.PATCH_SIZE[0], self.PATCH_SIZE[0], z_pad_size))[np.newaxis] # Must be 32 for ensembling!!!

            if self.dim==2:
                data_arr = np.transpose(data_arr,axes=(3, 0, 1, 2))
                seg_arr = np.transpose(seg_arr,axes=(3, 0, 1, 2))
            else:
                data_arr = data_arr[np.newaxis]
                seg_arr = seg_arr[np.newaxis]

            return {'data': data_arr.astype('float32'), 'seg': seg_arr.astype('float32'), 'pid': self.test_pid}





if __name__ == '__main__':

    import dm.nci_prostate.configs as cf
    import dm.nci_prostate.plotting
    print cf.patch_size
    batch_gen = get_train_generators(cf, fold=0)
    vbatch = next(batch_gen['val'])
    tbatch = next(batch_gen['train'])
    print vbatch['seg'].shape
    print tbatch['data'].shape
    print "last", tbatch['seg'].shape

    print "TESTING"
    test_data_dict = get_test_generator(cf)
    print test_data_dict[test_data_dict.keys()[0]]['data'].shape
    print test_data_dict[test_data_dict.keys()[0]]['seg'].shape


    # plotting.plot_batch_gen_example(next(batch_gen['train']), cf=cf, dim=cf.dim)
