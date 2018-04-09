# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# ---------------------------------------------------------
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import moving_averages


def linear(input_, output_size, stddev=0.02, bias_start=0.0, with_w=False, name='fc'):
    shape = input_.get_shape().as_list()
    # print('shape: ', shape)

    with tf.variable_scope(name) as scope:
        matrix = tf.get_variable(name="matrix", shape=[shape[1], output_size],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name="bias", shape=[output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def batch_norm(x, name, _ops, is_train=True):
    """Batch normalization."""
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable('beta', params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        if is_train is True:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                          initializer=tf.constant_initializer(0.0, tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)

            _ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
            _ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
            variance = tf.get_variable('moving_variance', params_shape, tf.float32, trainable=False)

        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)
        y.set_shape(x.get_shape())

        return y


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def upsampling2d(input_, size=(2, 2), name='upsampling2d'):
    with tf.name_scope(name):
        shape = input_.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(input_, size=(size[0] * shape[1], size[1] * shape[2]))


def max_pool_2x2(x, name='max_pool'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x, name=name)


def xavier_init(in_dim):
    print('in_dim: ', in_dim)
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return xavier_stddev


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

