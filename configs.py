__author__ = 'Paul F. Jaeger'

import os

#########################
#     IO Handling    #
#########################

experiment_name = 'try_dice_new_init'

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
patch_size=(288, 288)
n_workers = 10
n_cached = 10
worker_seeds=[123, 1234, 12345, 123456, 1234567, 12345678, 7, 8, 9, 10]
sample_void = False
data_aug_mode = 'train'

#########################
#       Training        #
#########################

n_epochs = 300
features_root = 30
n_cv_splits = 5
n_train_batches = 80
n_val_batches = 20
batch_size=10
slice_sample_thresh = 0.3

learning_rate = 10**(-3)
loss_name = 'dice_coefficient'
class_dict = {0:'bkgd', 1:'PZ', 2: "CG", 3: "FG"}

#########################
#       Testing        #
#########################
