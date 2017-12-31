__author__ = 'Paul F. Jaeger'

__author__ = 'Paul F. Jaeger'


import os
from configs_3D import *

#########################
#     IO Handling    #
#########################


root_dir = '/datasets/datasets_paul/dm/numpy_arrays/'
exp_dir='/home/jaegerp/experiments/tf_seg/{}'.format(experiment_name)
test_dir = os.path.join(exp_dir, 'test')
plot_dir = os.path.join(exp_dir, 'plots')

