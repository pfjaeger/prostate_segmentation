import numpy as np
import os
from batchgenerators.augmentations.utils import resize_image_by_padding, center_crop_2D_image_batched
from batchgenerators.dataloading.data_loader import DataLoaderBase
from batchgenerators.transforms.spatial_transforms import Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import TransposeChannels, ConvertSegToOnehotTransform
from sklearn.model_selection import KFold




def get_data_generators(cf, fold):


    fg = get_cv_fold_ixs(len_data=69)
    train_ix, val_ix = fg[fold]
    train_data = load_NCI_ISBI_dataset(cf, ids=train_ix)
    val_data = load_NCI_ISBI_dataset(cf, ids=val_ix)
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, do_aug=True)
    batch_gen['val'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)
    return batch_gen


def load_NCI_ISBI_dataset(cf, split='train', ids=()):

    in_dir = os.path.join(cf.root_dir, split)
    data = {}
    data_paths = [os.path.join(in_dir, f) for ix,f in enumerate(os.listdir(in_dir)) if (ix in ids) or len(ids)==0]
    concat_arr = [np.transpose(np.load(ii, mmap_mode='r'), axes=(2, 0, 1, 3)) for ii in data_paths]
    data['data'] = [ii[:, :, :, 0] for ii in concat_arr]
    data['seg'] =  [ii[:, :, :, 1] for ii in concat_arr]
    data['pid'] = [ii.split('/')[-1].split('.')[0] for ii in data_paths]
    return data


class BatchGenerator_2D(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE, PATCH_SIZE=(144, 144), n_batches=None):
        super(BatchGenerator_2D, self).__init__(data, BATCH_SIZE,  n_batches)
        self.PATCH_SIZE = PATCH_SIZE

    def generate_train_batch(self):

        patients = np.random.choice(range(len(self._data['pid'])), self.BATCH_SIZE, replace=True)
        data = []
        seg = []
        pids = []
        for nb in range(self.BATCH_SIZE):
            shp = self._data['data'][patients[nb]].shape
            slice_id = np.random.choice(shp[0])
            tmp_data = resize_image_by_padding(self._data['data'][patients[nb]][slice_id], (
            max(shp[1], self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])), pad_value=0)
            tmp_seg = resize_image_by_padding(self._data['seg'][patients[nb]][slice_id],
                                              (max(shp[1], self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])),
                                              pad_value=0)

            data.append(center_crop_2D_image_batched(tmp_data[np.newaxis, np.newaxis, :, :], self.PATCH_SIZE)[0])
            seg.append(center_crop_2D_image_batched(tmp_seg[np.newaxis, np.newaxis, :, :], self.PATCH_SIZE)[0])
            pids.append(self._data['pid'][patients[nb]])
        return {'data': np.array(data).astype('float32'), 'seg': np.array(seg).astype('float32'), 'pid': pids}


class TestGenerator_2D(DataLoaderBase):

    def __init__(self, data, BATCH_SIZE, PATCH_SIZE, test_ix, n_batches=None):
        super(TestGenerator_2D, self).__init__(data, BATCH_SIZE,  n_batches)
        self.PATCH_SIZE = PATCH_SIZE
        self.test_ix = test_ix

    def generate_train_batch(self):

            data = []
            seg = []
            pids = []
            shp = self._data['data'][self.test_ix].shape
            for slc in range(shp[0]):
                tmp_data = resize_image_by_padding(self._data['data'][self.test_ix][slc],
                                                   (max(shp[1], self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])), pad_value=0)
                tmp_seg = resize_image_by_padding(self._data['seg'][self.test_ix][slc],
                                                  (max(shp[1], self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])),pad_value=0)
                data.append(center_crop_2D_image_batched(tmp_data[np.newaxis, np.newaxis, :, :], self.PATCH_SIZE)[0])
                seg.append(center_crop_2D_image_batched(tmp_seg[np.newaxis, np.newaxis, :, :], self.PATCH_SIZE)[0])
                pids.append(self._data['pid'][self.test_ix])
            return {'data': np.array(data).astype('float32'), 'seg': np.array(seg).astype('float32'), 'pid': pids}


def create_data_gen_pipeline(patient_data, cf, test_ix=None, do_aug=True):

    if test_ix is None:
        data_gen = BatchGenerator_2D(patient_data, BATCH_SIZE=cf.batch_size, n_batches=None,
                                 PATCH_SIZE=cf.patch_size)

    else:
        data_gen = TestGenerator_2D(patient_data, BATCH_SIZE=cf.batch_size, n_batches=None,
                                 PATCH_SIZE=cf.patch_size, test_ix=test_ix)

    my_transforms = []
    if do_aug:
        mirror_transform = Mirror(axes=(2, 3))
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size, patch_center_dist_from_border=False,
                                             do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                                             do_rotation=True, angle_z=(0, 2 * np.pi),
                                             do_scale=True, scale=(0.8, 1.2),
                                             border_mode_data='constant', border_cval_data=0, order_data=1,
                                             random_crop=False)
        my_transforms.append(spatial_transform)

    my_transforms.append(ConvertSegToOnehotTransform(classes=(0, 1, 2)))
    my_transforms.append(TransposeChannels())

    all_transforms = Compose(my_transforms)
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator



def get_cv_fold_ixs(len_data):

    fold_list = []
    kf = KFold(n_splits=5, random_state=0, shuffle=True,)
    for train_index, val_index in kf.split(range(len_data)):
        fold_list.append([train_index, val_index])

    return fold_list