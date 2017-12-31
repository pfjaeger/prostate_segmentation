__author__ = 'Paul F. Jaeger'

import os

#########################
#     IO Handling    #
#########################

experiment_name = 'batch_dice'

root_dir = '/mnt/hdd/data/dm/numpy_arrays/'
exp_dir='/mnt/hdd/experiments/segmentation/{}'.format(experiment_name)
test_dir = os.path.join(exp_dir, 'test')
plot_dir = os.path.join(exp_dir, 'plots')


#########################
#      Data Loader      #
#########################


n_channels = 1
seg_option = 'lesion_detection' #classification / mal_detection / lesion_detection
task = 'classification'
n_classes=3



#########################
#    Batch Generation   #
#########################

seed=42
pad_size=(320, 320, 32)
patch_size=(288, 288, 32)
dim = 2
n_workers = 10
n_cached = 10
worker_seeds=[123, 1234, 12345, 123456, 1234567, 12345678, 7, 8, 9, 10]
sample_void = False
data_aug_mode = 'train'

#########################
#       Training        #
#########################

n_epochs = 300
features_root = 32
n_cv_splits = 5
n_train_batches = 80
n_val_batches = 20
batch_size=10
slice_sample_thresh = 0.0

learning_rate = 10**(-3) #-3 for wce super high beause loss downweighted.
loss_name = 'dice_coefficient'
class_weights = False
class_dict = {0:'bkgd', 1:'PZ', 2: 'CG', 3: 'FG'}


#########################
#   Data Augmentation   #
#########################

da_kwargs={
'rand_crop_dist': (pad_size[0]/2-10,pad_size[1]/2-10),
'do_elastic_deform': True,
'alpha':(0., 1500.),
'sigma':(30., 50.), #140 for 2d (288),
'do_rotation':True,
'angle_x': (0., 2* 3.14),
'angle_y': (0., 2* 3.14),
'angle_z': (0., 2* 3.14),
'do_scale':True,
'scale':(0.7, 1.3),
'random_crop':True}
