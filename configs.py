__author__ = 'Paul F. Jaeger'

import os

#########################
#   General Settings    #
#########################


# name of current experiment / output folder
experiment_name = 'test_merge'

# input and output directory paths
root_dir = '/mnt/hdd/data/dm/numpy_arrays/'
exp_dir='/mnt/hdd/experiments/segmentation/{}'.format(experiment_name) # DO JOIN HERE
test_dir = os.path.join(exp_dir, 'test_files')
plot_dir = os.path.join(exp_dir, 'plots')


# data dimensions
dim = 2
n_channels = 1
n_classes = 3

# seed for cross validation splits
seed=12345

# number of cpu used for multi-threaded batch generation
n_workers = 10

# training configs
n_cv_splits = 5
n_epochs = 300
learning_rate = 10**(-3)
loss_name = 'dice_coefficient'   # either one of 'cross_entropy', 'weighted_cross_entropy', or 'dice_coefficient'

# set class names for monitoring plot
class_dict = {0:'bkgd', 1:'PZ', 2: 'CG', 3: 'FG'}

# data Augmentation configs
da_kwargs={
'do_elastic_deform': True,
'alpha':(0., 1500.),
'sigma':(30., 50.),
'do_rotation':True,
'angle_z': (0., 2*3.14),
'do_scale':True,
'scale':(0.7, 1.3),
'random_crop':True,
}


#########################
#     2D   UNET        #
#########################

# settings specific to 2D UNet training.
if dim == 2:

    n_features_root = 32
    pre_crop_size = (320, 320)   #patch size before random_crop
    patch_size = (288, 288) #patch size after random_crop
    network_input_shape = [None, patch_size[0], patch_size[1], n_channels]
    network_output_shape = [None, patch_size[0], patch_size[1], n_classes]

    n_train_batches = 80
    n_val_batches = 20
    batch_size= 10
    da_kwargs['angle_x'] = (0, 2*3.14)
    da_kwargs['angle_y'] = (0, 2*3.14)

    # allowed shift from the patch center during random crop
    da_kwargs['rand_crop_dist'] = (pre_crop_size[0] / 2. - 5, pre_crop_size[1] / 2. - 5)


#########################
#        3D UNET        #
#########################

# settings specific to 3D UNet training.
if dim == 3:

    n_features_root = 12
    pre_crop_size = (320, 320, 32)  # patch size before random_crop
    patch_size = (288, 288, 32) #patch size after random_crop
    network_input_shape = [None, patch_size[2], patch_size[0], patch_size[1], n_channels]
    network_output_shape = [None, patch_size[2], patch_size[0], patch_size[1], n_classes]

    n_train_batches = 20
    n_val_batches = 5
    batch_size = 2
    da_kwargs['angle_x'] = (0, 0.05)    # only allow for slight tilt along x and y axis.
    da_kwargs['angle_y'] = (0, 0.05)

    # allowed shift from the patch center during random crop
    da_kwargs['rand_crop_dist'] = (pre_crop_size[0] / 2. - 5, pre_crop_size[1] / 2. - 5, pre_crop_size[2] / 2. - 2)

