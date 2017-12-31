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




def get_train_generators(cf, fold):

    fg = get_cv_fold_ixs(len_data=69)
    train_ix, val_ix = fg[fold]
    train_data = load_NCI_ISBI_dataset(cf, ids=train_ix)
    val_data = load_NCI_ISBI_dataset(cf, ids=val_ix)
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, do_aug=True)
    batch_gen['val'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)
    return batch_gen


def get_test_generator(cf):

    test_data = load_NCI_ISBI_dataset(cf, split='test')
    test_data_dict = {}
    for ix, pid in enumerate(test_data['pid']):
        test_gen = create_data_gen_pipeline(test_data, cf=cf, test_ix=ix, do_aug=False)
        test_data_dict[pid] = next(test_gen)
    return test_data_dict

def load_NCI_ISBI_dataset(cf, split='train', ids=()):

    in_dir = os.path.join(cf.root_dir, split)
    data = {}
    data_paths = [os.path.join(in_dir, f) for ix,f in enumerate(os.listdir(in_dir)) if (ix in ids) or len(ids)==0]
    concat_arr = [np.load(ii, mmap_mode='r') for ii in data_paths]
    #DELTE!! DEPRECATED SLICE IXS
    # relevant_slices = [np.unique(np.argwhere(arr[:, :, :, 1] != 0)[:, 0]) for arr in concat_arr]
    # concat_arr = [arr[np.min(relevant_slices[ix]): np.max(relevant_slices[ix]), :, :, :] for ix, arr in enumerate(concat_arr)]

    data['data'] = [ii[:, :, :, 0] for ii in concat_arr]
    data['seg'] =  [ii[:, :, :, 1] for ii in concat_arr]
    data['pid'] = [ii.split('/')[-1].split('.')[0] for ii in data_paths]
    return data


def get_cv_fold_ixs(len_data):

    fold_list = []
    kf = KFold(n_splits=5, random_state=0, shuffle=True,)
    for train_index, val_index in kf.split(range(len_data)):
        fold_list.append([train_index, val_index])

    return fold_list


def create_data_gen_pipeline(patient_data, cf, test_ix=None, do_aug=True):

    if cf.dim==2:
        if test_ix is None:
            data_gen = BatchGenerator_2D(patient_data, BATCH_SIZE=cf.batch_size, n_batches=None,
                                     PATCH_SIZE=cf.pad_size, slice_sample_thresh=cf.slice_sample_thresh, do_aug=do_aug)

        else:
            data_gen = TestGenerator(patient_data, BATCH_SIZE=cf.batch_size, n_batches=None,
                                     PATCH_SIZE=cf.pad_size, test_ix=test_ix)

    else:
        if test_ix is None:
            data_gen = BatchGenerator_3D(patient_data, BATCH_SIZE=cf.batch_size, n_batches=None,
                                         PATCH_SIZE=cf.pad_size, slice_sample_thresh=cf.slice_sample_thresh,
                                         do_aug=do_aug)

        else:
            data_gen = TestGenerator(patient_data, BATCH_SIZE=cf.batch_size, n_batches=None,
                                        PATCH_SIZE=cf.pad_size, test_ix=test_ix, dim=3)

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


class BatchGenerator_2D(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE, PATCH_SIZE, n_batches=None, slice_sample_thresh=0.2, do_aug=False):
        super(BatchGenerator_2D, self).__init__(data, BATCH_SIZE,  n_batches)
        self.PATCH_SIZE = PATCH_SIZE
        self.slice_sample_thresh = slice_sample_thresh
        self.do_aug = do_aug

    def generate_train_batch(self):

        patients = np.random.choice(range(len(self._data['pid'])), self.BATCH_SIZE, replace=True)
        data = []
        seg = []
        pids = []
        for b in range(self.BATCH_SIZE):
            shp = self._data['data'][patients[b]].shape

            if self.do_aug and 1==0: #DELETE!
                #importance sampling
                is_filled = [1 if np.sum(self._data['seg'][patients[b]][:, :, ix]!=0)>0 else 0 for ix in range(shp[2])]
                filled_slice_ixs = [ix for ix, ii in enumerate(is_filled) if ii==1]
                empty_slice_ixs = [ix for ix, ii in enumerate(is_filled) if ii==0]
                sample = np.random.uniform()
                if sample > self.slice_sample_thresh or len(empty_slice_ixs)==0:
                        slice_ix = np.random.choice(filled_slice_ixs)
                else:
                        slice_ix = np.random.choice(empty_slice_ixs)
            else:
                slice_ix = np.random.choice(range(shp[2]))
            tmp_data = resize_image_by_padding(self._data['data'][patients[b]][:, :, slice_ix], (
            max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1])), pad_value=0)
            tmp_seg = resize_image_by_padding(self._data['seg'][patients[b]][:, :, slice_ix],
                                              (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1])),
                                              pad_value=0)
            data.append(center_crop_2D_image(tmp_data, self.PATCH_SIZE[:2])[np.newaxis])
            seg.append(center_crop_2D_image(tmp_seg, self.PATCH_SIZE[:2])[np.newaxis])
            pids.append(self._data['pid'][patients[b]])
        return {'data': np.array(data).astype('float32'), 'seg': np.array(seg).astype('float32'), 'pid': pids}


class TestGenerator(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE, PATCH_SIZE, test_ix, n_batches=None, dim=2):
        super(TestGenerator, self).__init__(data, BATCH_SIZE,  n_batches)
        self.PATCH_SIZE = PATCH_SIZE
        self.test_ix = test_ix
        self.dim = dim

    def generate_train_batch(self): #naming!?

            data = []
            seg = []
            pids = []
            shp = self._data['data'][self.test_ix].shape
            #actually i dont need the padding since all imgs are bigger than patch size.
            #no padding cropping in z. becuase testing! do this in 3D sincein train and test 1 batch is 1 patient volume.
            tmp_data = resize_image_by_padding(self._data['data'][self.test_ix],
                                               (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), max(shp[2], self.PATCH_SIZE[2])), pad_value=0)
            tmp_seg = resize_image_by_padding(self._data['seg'][self.test_ix],
                                               (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), max(shp[2], self.PATCH_SIZE[2])), pad_value=0)
            data.append(center_crop_3D_image(tmp_data, self.PATCH_SIZE))
            seg.append(center_crop_3D_image(tmp_seg, self.PATCH_SIZE))
            pids.append(self._data['pid'][self.test_ix])
            if self.dim==2:
                data_arr = np.transpose(np.array(data).astype('float32'),axes=(3, 0, 1, 2))
                seg_arr = np.transpose(np.array(seg).astype('float32'),axes=(3, 0, 1, 2))
            else:
                data_arr = np.array(data).astype('float32')[np.newaxis]
                seg_arr = np.array(seg).astype('float32')[np.newaxis]
            return {'data': data_arr, 'seg': seg_arr, 'pid': pids}



class BatchGenerator_3D(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE, PATCH_SIZE=(144, 144, 32), n_batches=None, slice_sample_thresh=0.2, do_aug=False):
        super(BatchGenerator_3D, self).__init__(data, BATCH_SIZE,  n_batches)
        self.PATCH_SIZE = PATCH_SIZE
        self.slice_sample_thresh = slice_sample_thresh
        self.do_aug = do_aug

    def generate_train_batch(self):

        patients = np.random.choice(range(len(self._data['pid'])), self.BATCH_SIZE, replace=True)
        data = []
        seg = []
        pids = []
        for b in range(self.BATCH_SIZE):
            shp = self._data['data'][patients[b]].shape

            tmp_data = resize_image_by_padding(self._data['data'][patients[b]], (
            max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), max(shp[2], self.PATCH_SIZE[2])), pad_value=0)
            tmp_seg = resize_image_by_padding(self._data['seg'][patients[b]],
                                              (max(shp[0], self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), max(shp[2], self.PATCH_SIZE[2])),
                                              pad_value=0)
            data.append(center_crop_3D_image(tmp_data, self.PATCH_SIZE)[np.newaxis])
            seg.append(center_crop_3D_image(tmp_seg, self.PATCH_SIZE)[np.newaxis])
            pids.append(self._data['pid'][patients[b]])
        return {'data': np.array(data).astype('float32'), 'seg': np.array(seg).astype('float32'), 'pid': pids}



if __name__ == '__main__':

    import configs_3D as cf
    import plotting
    print cf.patch_size
    batch_gen = get_train_generators(cf, fold=0)
    # vbatch = next(batch_gen['val'])
    tbatch = next(batch_gen['train'])
    # test_data_dict = get_test_generator(cf)
    # print test_data_dict.keys()
    # print vbatch['seg'].shape
    print tbatch['data'].shape
    # print "last",tbatch['seg'].shape
    plotting.plot_batch_gen_example(next(batch_gen['train']), cf=cf, dim=cf.dim)
