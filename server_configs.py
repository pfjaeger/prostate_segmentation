__author__ = 'Paul F. Jaeger'


import os
from configs import *

#########################
#     IO Handling    #
#########################

experiment_name = 'lean_script_tf'

root_dir = '/datasets/datasets/paul/dm/numpy_arrays/'
exp_dir='/home/experiments/tf_seg/{}'.format(experiment_name)
test_dir = os.path.join(exp_dir, 'test')
plot_dir = os.path.join(exp_dir, 'plots')

