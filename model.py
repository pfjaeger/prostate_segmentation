__author__ = 'Paul F. Jaeger'

import tensorflow as tf
from collections import OrderedDict
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import instance_norm
from tensorflow.contrib.layers.python.layers import initializers

def leaky_relu(x):
    half_alpha = 0.01
    return (0.5 + half_alpha) * x + (0.5 - half_alpha) * abs(x)


def create_UNet(x, features_root=16, num_classes=2):

    net = OrderedDict()

    # down layers # eliminate scop declaration if not using tensorboard
    net['encode/conv1'] = slim.repeat(x, 2, slim.conv2d, features_root, [3, 3], scope='encode/conv1')
    net['encode/pool1'] = slim.max_pool2d(net['encode/conv1'], [2, 2], scope='encode/pool1')
    net['encode/conv2'] = slim.repeat(net['encode/pool1'], 2, slim.conv2d, features_root * 2, [3, 3], scope='encode/conv2')
    net['encode/pool2'] = slim.max_pool2d(net['encode/conv2'], [2, 2], scope='encode/pool2')
    net['encode/conv3'] = slim.repeat(net['encode/pool2'], 2, slim.conv2d, features_root * 4, [3, 3], scope='encode/conv3')
    net['encode/pool3'] = slim.max_pool2d(net['encode/conv3'], [2, 2], scope='encode/pool3')
    net['encode/conv4'] = slim.repeat(net['encode/pool3'], 2, slim.conv2d, features_root * 8, [3, 3], scope='encode/conv4')
    net['encode/pool4'] = slim.max_pool2d(net['encode/conv4'], [2, 2], scope='encode/pool4')
    net['encode/conv5'] = slim.repeat(net['encode/pool4'], 2, slim.conv2d, features_root * 16, [3, 3], scope='encode/conv5')

    net['decode/up_conv1'] = slim.conv2d_transpose(net['encode/conv5'], features_root * 8, 2, stride=2, activation_fn=None, padding='VALID')
    net['decode/concat_c4_u1'] = tf.concat([net['encode/conv4'], net['decode/up_conv1']], 3)
    net['decode/conv1'] = slim.repeat(net['decode/concat_c4_u1'], 2, slim.conv2d, features_root * 8, [3, 3], scope='decode/conv1')
    net['decode/up_conv2'] = slim.conv2d_transpose(net['decode/conv1'], features_root * 4, 2, stride=2, activation_fn=None, padding='VALID')
    net['decode/concat_c3_u2'] = tf.concat([net['encode/conv3'], net['decode/up_conv2']], 3)
    net['decode/conv2'] = slim.repeat(net['decode/concat_c3_u2'], 2, slim.conv2d, features_root * 4, [3, 3], scope='decode/conv2')
    net['decode/up_conv3'] = slim.conv2d_transpose(net['decode/conv2'], features_root * 2, 2, stride=2, activation_fn=None, padding='VALID')
    net['decode/concat_c2_u3'] = tf.concat([net['encode/conv2'], net['decode/up_conv3']], 3)
    net['decode/conv3'] = slim.repeat(net['decode/concat_c2_u3'], 2, slim.conv2d, features_root * 2, [3, 3], scope='decode/conv3')
    net['decode/up_conv4'] = slim.conv2d_transpose(net['decode/conv3'], features_root, 2, stride=2, activation_fn=None, padding='VALID')
    net['decode/concat_c1_u4'] = tf.concat([net['encode/conv1'], net['decode/up_conv4']], 3)
    net['decode/conv4'] = slim.repeat(net['decode/concat_c1_u4'], 2, slim.conv2d, features_root, [3, 3], scope='decode/conv4')


    net['out_map'] = slim.conv2d(net['decode/conv4'], num_classes, [1, 1], activation_fn=None)

    for i, c in net.iteritems():
        print(i, c.get_shape().as_list())

    return net['out_map'], tf.global_variables()



def create_BN_UNet(x, features_root=16, num_classes=2, is_training=True):

    net = OrderedDict()

    net['encode/conv1'] = batch_norm(slim.repeat(x, 2, slim.conv2d, features_root, [3, 3], scope='encode/conv1'), is_training=is_training)
    net['encode/pool1'] = slim.max_pool2d(net['encode/conv1'], [2, 2], scope='encode/pool1')
    net['encode/conv2'] = batch_norm(
        slim.repeat(net['encode/pool1'], 2, slim.conv2d, features_root * 2, [3, 3], scope='encode/conv2'), is_training=is_training)
    net['encode/pool2'] = slim.max_pool2d(net['encode/conv2'], [2, 2], scope='encode/pool2')
    net['encode/conv3'] = batch_norm(
        slim.repeat(net['encode/pool2'], 2, slim.conv2d, features_root * 4, [3, 3], scope='encode/conv3'), is_training=is_training)
    net['encode/pool3'] = slim.max_pool2d(net['encode/conv3'], [2, 2], scope='encode/pool3')
    net['encode/conv4'] = batch_norm(
        slim.repeat(net['encode/pool3'], 2, slim.conv2d, features_root * 8, [3, 3], scope='encode/conv4'), is_training=is_training)
    net['encode/pool4'] = slim.max_pool2d(net['encode/conv4'], [2, 2], scope='encode/pool4')
    net['encode/conv5'] = batch_norm(
        slim.repeat(net['encode/pool4'], 2, slim.conv2d, features_root * 16, [3, 3], scope='encode/conv5'), is_training=is_training)

    net['decode/up_conv1'] = slim.conv2d_transpose(net['encode/conv5'], features_root * 8, 2, stride=2,
                                                   activation_fn=None, padding='VALID')
    net['decode/concat_c4_u1'] = tf.concat([net['encode/conv4'], net['decode/up_conv1']], 3)
    net['decode/conv1'] = batch_norm(
        slim.repeat(net['decode/concat_c4_u1'], 2, slim.conv2d, features_root * 8, [3, 3], scope='decode/conv1'), is_training=is_training)
    net['decode/up_conv2'] = slim.conv2d_transpose(net['decode/conv1'], features_root * 4, 2, stride=2,
                                                   activation_fn=None, padding='VALID')
    net['decode/concat_c3_u2'] = tf.concat([net['encode/conv3'], net['decode/up_conv2']], 3)
    net['decode/conv2'] = batch_norm(
        slim.repeat(net['decode/concat_c3_u2'], 2, slim.conv2d, features_root * 4, [3, 3], scope='decode/conv2'), is_training=is_training)
    net['decode/up_conv3'] = slim.conv2d_transpose(net['decode/conv2'], features_root * 2, 2, stride=2,
                                                   activation_fn=None, padding='VALID')
    net['decode/concat_c2_u3'] = tf.concat([net['encode/conv2'], net['decode/up_conv3']], 3)
    net['decode/conv3'] = batch_norm(
        slim.repeat(net['decode/concat_c2_u3'], 2, slim.conv2d, features_root * 2, [3, 3], scope='decode/conv3'), is_training=is_training)
    net['decode/up_conv4'] = slim.conv2d_transpose(net['decode/conv3'], features_root, 2, stride=2, activation_fn=None,
                                                   padding='VALID')
    net['decode/concat_c1_u4'] = tf.concat([net['encode/conv1'], net['decode/up_conv4']], 3)
    net['decode/conv4'] = batch_norm(
        slim.repeat(net['decode/concat_c1_u4'], 2, slim.conv2d, features_root, [3, 3], scope='decode/conv4'), is_training=is_training)


    net['out_map'] = batch_norm(slim.conv2d(net['decode/conv4'], num_classes, [1, 1], activation_fn=None), is_training=is_training)

    for i, c in net.iteritems():
        print(i, c.get_shape().as_list())

    return net['out_map'], tf.global_variables()





def create_IN_UNet(x, features_root=16, num_classes=2, is_training=True):

    net = OrderedDict()

    net['encode/conv1'] = instance_norm(slim.repeat(x, 2, slim.conv2d, features_root, [3, 3], scope='encode/conv1'))
    net['encode/pool1'] = slim.max_pool2d(net['encode/conv1'], [2, 2], scope='encode/pool1')
    net['encode/conv2'] = instance_norm(
        slim.repeat(net['encode/pool1'], 2, slim.conv2d, features_root * 2, [3, 3], scope='encode/conv2'))
    net['encode/pool2'] = slim.max_pool2d(net['encode/conv2'], [2, 2], scope='encode/pool2')
    net['encode/conv3'] = instance_norm(
        slim.repeat(net['encode/pool2'], 2, slim.conv2d, features_root * 4, [3, 3], scope='encode/conv3'))
    net['encode/pool3'] = slim.max_pool2d(net['encode/conv3'], [2, 2], scope='encode/pool3')
    net['encode/conv4'] = instance_norm(
        slim.repeat(net['encode/pool3'], 2, slim.conv2d, features_root * 8, [3, 3], scope='encode/conv4'))
    net['encode/pool4'] = slim.max_pool2d(net['encode/conv4'], [2, 2], scope='encode/pool4')
    net['encode/conv5'] = instance_norm(
        slim.repeat(net['encode/pool4'], 2, slim.conv2d, features_root * 16, [3, 3], scope='encode/conv5'))

    net['decode/up_conv1'] = slim.conv2d_transpose(net['encode/conv5'], features_root * 8, 2, stride=2,
                                                   activation_fn=None, padding='VALID')
    net['decode/concat_c4_u1'] = tf.concat([net['encode/conv4'], net['decode/up_conv1']], 3)
    net['decode/conv1'] = instance_norm(
        slim.repeat(net['decode/concat_c4_u1'], 2, slim.conv2d, features_root * 8, [3, 3], scope='decode/conv1'))
    net['decode/up_conv2'] = slim.conv2d_transpose(net['decode/conv1'], features_root * 4, 2, stride=2,
                                                   activation_fn=None, padding='VALID')
    net['decode/concat_c3_u2'] = tf.concat([net['encode/conv3'], net['decode/up_conv2']], 3)
    net['decode/conv2'] = instance_norm(
        slim.repeat(net['decode/concat_c3_u2'], 2, slim.conv2d, features_root * 4, [3, 3], scope='decode/conv2'))
    net['decode/up_conv3'] = slim.conv2d_transpose(net['decode/conv2'], features_root * 2, 2, stride=2,
                                                   activation_fn=None, padding='VALID')
    net['decode/concat_c2_u3'] = tf.concat([net['encode/conv2'], net['decode/up_conv3']], 3)
    net['decode/conv3'] = instance_norm(
        slim.repeat(net['decode/concat_c2_u3'], 2, slim.conv2d, features_root * 2, [3, 3], scope='decode/conv3'))
    net['decode/up_conv4'] = slim.conv2d_transpose(net['decode/conv3'], features_root, 2, stride=2, activation_fn=None,
                                                   padding='VALID')
    net['decode/concat_c1_u4'] = tf.concat([net['encode/conv1'], net['decode/up_conv4']], 3)
    net['decode/conv4'] = instance_norm(
        slim.repeat(net['decode/concat_c1_u4'], 2, slim.conv2d, features_root, [3, 3], scope='decode/conv4'))


    net['out_map'] = instance_norm(slim.conv2d(net['decode/conv4'], num_classes, [1, 1], activation_fn=None))

    for i, c in net.iteritems():
        print(i, c.get_shape().as_list())

    return net['out_map'], tf.global_variables()




def create_nice_UNet(x, features_root=16, n_classes=2, is_training=True):

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

    for i, c in net.iteritems():
        print(i, c.get_shape().as_list())

    return net['out_map'], tf.global_variables()







def create_3D_UNet(x, features_root=16, n_classes=2, is_training=True):

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

    for i, c in net.iteritems():
        print(i, c.get_shape().as_list())

    return net['out_map'], tf.global_variables()
