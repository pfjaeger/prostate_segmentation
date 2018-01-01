__author__ = 'Paul F. Jaeger'


import os

#########################
#     IO Handling    #
#########################

experiment_name = 'test_merge'

root_dir = '/mnt/hdd/data/dm/numpy_arrays/'
exp_dir='/mnt/hdd/experiments/segmentation/{}'.format(experiment_name)
test_dir = os.path.join(exp_dir, 'test')
plot_dir = os.path.join(exp_dir, 'plots')



#Data Dimensions
dim = 2
n_channels = 1
n_classes=3

seed=12345
n_workers = 10

# Training configs
n_cv_splits = 5
n_epochs = 300
learning_rate = 10**(-3) #-3 for wce super high beause loss downweighted.
loss_name = 'dice_coefficient'
class_weights = False
class_dict = {0:'bkgd', 1:'PZ', 2: 'CG', 3: 'FG'}

# Data Augmentation configs
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

if dim == 2:

    n_features_root = 32
    pad_size = (320, 320)
    patch_size = (288, 288)
    network_input_shape = [None, patch_size[0], patch_size[1], n_channels]
    network_output_shape = [None, patch_size[0], patch_size[1], n_classes]

    n_train_batches = 80
    n_val_batches = 20
    batch_size= 10
    da_kwargs['angle_x'] = (0, 2*3.14)
    da_kwargs['angle_y'] = (0, 2*3.14)
    da_kwargs['rand_crop_dist'] = (pad_size[0] / 2. - 5, pad_size[1] / 2. - 5)


#########################
#        3D UNET        #
#########################

if dim == 3:

    n_features_root = 12
    patch_size = (288, 288, 32)
    pad_size = (320, 320, 32)
    network_input_shape = [None, patch_size[2], patch_size[0], patch_size[1], n_channels]
    network_output_shape = [None, patch_size[2], patch_size[0], patch_size[1], n_classes]

    n_train_batches = 20
    n_val_batches = 5
    batch_size = 2
    da_kwargs['angle_x'] = (0, 0.05)
    da_kwargs['angle_y'] = (0, 0.05)
    da_kwargs['rand_crop_dist'] = (pad_size[0] / 2. - 5, pad_size[1] / 2. - 5, pad_size[2] / 2. - 2)

