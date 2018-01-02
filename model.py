__author__ = 'Paul F. Jaeger'

import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict
from tensorflow.contrib.layers import instance_norm
from tensorflow.contrib.layers.python.layers import initializers


def create_UNet(x, features_root, n_classes, dim, logger):

    if dim == 2:
        net, variables = create_2D_UNet(x, features_root, n_classes)
    elif dim == 3:
        net, variables = create_3D_UNet(x, features_root, n_classes)
    else:
        raise ValueError("wrong dimension selected in configs.")

    for i, c in net.iteritems():
        print(i, c.get_shape().as_list())

    return net, variables


def leaky_relu(x):
    """
    from https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/activation.py
    """
    half_alpha = 0.01
    return (0.5 + half_alpha) * x + (0.5 - half_alpha) * abs(x)


def create_2D_UNet(x, features_root, n_classes):

    net = OrderedDict()
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                   weights_initializer = initializers.variance_scaling_initializer(
                       factor=2.0, mode='FAN_IN', uniform=False), activation_fn=leaky_relu):

        net['encode/conv1_1'] = instance_norm(slim.conv2d(x, features_root, [3, 3]))
        net['encode/conv1_2'] = instance_norm(slim.conv2d(net['encode/conv1_1'], features_root, [3, 3]))
        net['encode/pool1'] = slim.max_pool2d(net['encode/conv1_2'], [2, 2])

        net['encode/conv2_1'] = instance_norm(slim.conv2d(net['encode/pool1'], features_root*2, [3, 3]))
        net['encode/conv2_2'] = instance_norm(slim.conv2d(net['encode/conv2_1'], features_root*2, [3, 3]))
        net['encode/pool2'] = slim.max_pool2d(net['encode/conv2_2'], [2, 2])

        net['encode/conv3_1'] = instance_norm(slim.conv2d(net['encode/pool2'], features_root*4, [3, 3]))
        net['encode/conv3_2'] = instance_norm(slim.conv2d(net['encode/conv3_1'], features_root*4, [3, 3]))
        net['encode/pool3'] = slim.max_pool2d(net['encode/conv3_2'], [2, 2])

        net['encode/conv4_1'] = instance_norm(slim.conv2d(net['encode/pool3'], features_root*8, [3, 3]))
        net['encode/conv4_2'] = instance_norm(slim.conv2d(net['encode/conv4_1'], features_root*8, [3, 3]))
        net['encode/pool4'] = slim.max_pool2d(net['encode/conv4_2'], [2, 2])

        net['encode/conv5_1'] = instance_norm(slim.conv2d(net['encode/pool4'], features_root*16, [3, 3]))
        net['encode/conv5_2'] = instance_norm(slim.conv2d(net['encode/conv5_1'], features_root*16, [3, 3]))

        net['decode/up_conv1'] = slim.conv2d_transpose(net['encode/conv5_2'], features_root * 8, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c4_u1'] = tf.concat([net['encode/conv4_2'], net['decode/up_conv1']], 3)
        net['decode/conv1_1'] = instance_norm(slim.conv2d(net['decode/concat_c4_u1'], features_root * 8, [3, 3]))
        net['decode/conv1_2'] = instance_norm(slim.conv2d(net['decode/conv1_1'], features_root * 8, [3, 3]))

        net['decode/up_conv2'] = slim.conv2d_transpose(net['decode/conv1_2'], features_root * 4, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c3_u2'] = tf.concat([net['encode/conv3_2'], net['decode/up_conv2']], 3)
        net['decode/conv2_1'] = instance_norm(slim.conv2d(net['decode/concat_c3_u2'], features_root * 4, [3, 3]))
        net['decode/conv2_2'] = instance_norm(slim.conv2d(net['decode/conv2_1'], features_root * 4, [3, 3]))

        net['decode/up_conv3'] = slim.conv2d_transpose(net['decode/conv2_2'], features_root * 2, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c2_u3'] = tf.concat([net['encode/conv2_2'], net['decode/up_conv3']], 3)
        net['decode/conv3_1'] = instance_norm(slim.conv2d(net['decode/concat_c2_u3'], features_root * 2, [3, 3]))
        net['decode/conv3_2'] = instance_norm(slim.conv2d(net['decode/conv3_1'], features_root * 2, [3, 3]))

        net['decode/up_conv4'] = slim.conv2d_transpose(net['decode/conv3_2'], features_root, 2,
                                                       stride=2, activation_fn=None, padding='VALID')
        net['decode/concat_c1_u4'] = tf.concat([net['encode/conv1_2'], net['decode/up_conv4']], 3)
        net['decode/conv4_1'] = instance_norm(slim.conv2d(net['decode/concat_c1_u4'], features_root, [3, 3]))
        net['decode/conv4_2'] = instance_norm(slim.conv2d(net['decode/conv4_1'], features_root, [3, 3]))

        net['out_map'] = instance_norm(slim.conv2d(net['decode/conv4_2'], n_classes, [1, 1], activation_fn=None))

    return net['out_map'], tf.global_variables()


def create_3D_UNet(x, features_root=16, n_classes=2):

    net = OrderedDict()
    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                   weights_initializer = initializers.variance_scaling_initializer(
                       factor=2.0, mode='FAN_IN', uniform=False), activation_fn=leaky_relu):

        net['encode/conv1_1'] = instance_norm(slim.conv3d(x, features_root, [3, 3, 3]))
        net['encode/conv1_2'] = instance_norm(slim.conv3d(net['encode/conv1_1'], features_root, [3, 3, 3]))
        net['encode/pool1'] = slim.max_pool3d(net['encode/conv1_2'], kernel_size=[1, 2, 2], stride=[1,2,2])

        net['encode/conv2_1'] = instance_norm(slim.conv3d(net['encode/pool1'], features_root*2, [3, 3, 3]))
        net['encode/conv2_2'] = instance_norm(slim.conv3d(net['encode/conv2_1'], features_root*2, [3, 3, 3]))
        net['encode/pool2'] = slim.max_pool3d(net['encode/conv2_2'], kernel_size=[2, 2, 2], stride=[2,2,2])

        net['encode/conv3_1'] = instance_norm(slim.conv3d(net['encode/pool2'], features_root*4, [3, 3, 3]))
        net['encode/conv3_2'] = instance_norm(slim.conv3d(net['encode/conv3_1'], features_root*4, [3, 3, 3]))
        net['encode/pool3'] = slim.max_pool3d(net['encode/conv3_2'], [2, 2, 2])

        net['encode/conv4_1'] = instance_norm(slim.conv3d(net['encode/pool3'], features_root*8, [3, 3, 3]))
        net['encode/conv4_2'] = instance_norm(slim.conv3d(net['encode/conv4_1'], features_root*8, [3, 3, 3]))
        net['encode/pool4'] = slim.max_pool3d(net['encode/conv4_2'], [2, 2, 2])

        net['encode/conv5_1'] = instance_norm(slim.conv3d(net['encode/pool4'], features_root*16, [3, 3, 3]))
        net['encode/conv5_2'] = instance_norm(slim.conv3d(net['encode/conv5_1'], features_root*16, [3, 3, 3]))

        net['decode/up_conv1'] = slim.conv3d_transpose(net['encode/conv5_2'], features_root * 8, [2, 2, 2],
                                                       stride=2, activation_fn=None, padding='VALID', biases_initializer=None)
        net['decode/concat_c4_u1'] = tf.concat([net['encode/conv4_2'], net['decode/up_conv1']], 4)
        net['decode/conv1_1'] = instance_norm(slim.conv3d(net['decode/concat_c4_u1'], features_root * 8, [3, 3, 3]))
        net['decode/conv1_2'] = instance_norm(slim.conv3d(net['decode/conv1_1'], features_root * 8, [3, 3, 3]))


        net['decode/up_conv2'] = slim.conv3d_transpose(net['decode/conv1_2'], features_root * 4, [2, 2, 2],
                                                       stride=2, activation_fn=None, padding='VALID', biases_initializer=None)


        net['decode/concat_c3_u2'] = tf.concat([net['encode/conv3_2'], net['decode/up_conv2']], 4)
        net['decode/conv2_1'] = instance_norm(slim.conv3d(net['decode/concat_c3_u2'], features_root * 4, [3, 3, 3]))
        net['decode/conv2_2'] = instance_norm(slim.conv3d(net['decode/conv2_1'], features_root * 4, [3, 3, 3]))

        net['decode/up_conv3'] = slim.conv3d_transpose(net['decode/conv2_2'], features_root * 2, kernel_size=[2, 2, 2], stride=[2,2,2],
                                                       activation_fn=None, padding='VALID', biases_initializer=None)
        net['decode/concat_c2_u3'] = tf.concat([net['encode/conv2_2'], net['decode/up_conv3']], 4)
        net['decode/conv3_1'] = instance_norm(slim.conv3d(net['decode/concat_c2_u3'], features_root * 2, [3, 3, 3]))
        net['decode/conv3_2'] = instance_norm(slim.conv3d(net['decode/conv3_1'], features_root * 2, [3, 3, 3]))

        net['decode/up_conv4'] = slim.conv3d_transpose(net['decode/conv3_2'], features_root,  [1, 2, 2],
                                                       stride=[1, 2, 2], activation_fn=None, padding='VALID', biases_initializer=None)

        net['decode/concat_c1_u4'] = tf.concat([net['encode/conv1_2'], net['decode/up_conv4']], 4)
        net['decode/conv4_1'] = instance_norm(slim.conv3d(net['decode/concat_c1_u4'], features_root, [3, 3, 3]))
        net['decode/conv4_2'] = instance_norm(slim.conv3d(net['decode/conv4_1'], features_root, [3, 3, 3]))

        net['out_map'] = instance_norm(slim.conv3d(net['decode/conv4_2'], n_classes, [1, 1, 1], activation_fn=None))

    return net['out_map'], tf.global_variables()
