#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:25:24 2019

@author: xulin
"""

import tensorflow as tf
slim = tf.contrib.slim

"""
    Encoder
"""

# Create a variable.
def create_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.997)
    return tf.get_variable(name, shape = shape, initializer = initializer, regularizer = regularizer)

# Create a stride.
def create_stride(stride):
    return [1, stride, stride, 1]

# Convolutional layer
def conv_layer(name, x, filter_shape, stride = create_stride(1), pad = "SAME"):
    # x: the input layer
    # filter_shape: a 4d vector, [filter_height, filter_width, filter_depth, filter_number]
    # stride: the stride on the height and width
    # pad: the padding variable in tf.nn.conv2d
    return tf.nn.conv2d(input = x, filter = filter_shape, strides = stride, padding = pad, name = name)

# Max pooling layer
def pooling_layer(name, x, pooling_size, stride = create_stride(2), pad = "SAME"):
    return tf.nn.max_pool(value = x, ksize = pooling_size, strides = stride, padding = pad, name = name)

# Flatten layer
def flatten_layer(x):
    shape = x.get_shape().as_list()
    return tf.reshape(x, [shape[0], -1])

# Fully connected layer
def fully_connected_layer(x, n_out, namew, nameb):
    shape = x.get_shape().as_list()
    n_in = shape[-1]
    fcw = create_variable(name = namew, shape = [n_in, n_out], initializer = tf.uniform_unit_scaling_initializer(factor = 1.0))
    fcb = create_variable(name = nameb, shape = [n_out], initializer = tf.truncated_normal_initializer())
    
#    fcw = tf.get_variable(name = namew, shape = [n_in, n_out], initializer = tf.uniform_unit_scaling_initializer(factor = 1.0))
#    fcb = tf.get_variable(name = nameb, shape = [n_out], initializer = tf.truncated_normal_initializer())
    return tf.nn.xw_plus_b(x, fcw, fcb)

# Leaky relu layer
def relu_layer(x):
    return tf.nn.leaky_relu(x, alpha = 0.1)

# Batch normalization layer
def batch_normalization(x, name):
    mean, variance = tf.nn.moments(x, axes = [0, 1, 2], keep_dims = False)
    dimension = x.get_shape().as_list()[-1]
    beta = create_variable(name = name + "_beta", shape = [dimension], initializer = tf.constant_initializer(1.0, tf.float32))
    gamma = create_variable(name = name + "_gamma", shape = [dimension], initializer = tf.constant_initializer(1.0, tf.float32))
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)


"""
    GRU
"""

# fcconv3d layer
def fcconv3d_layer(h_t, feature_x, filters, n_gru_vox, namew, nameb):
    out_shape = h_t.get_shape().as_list()
    fc_output = fully_connected_layer(feature_x, n_gru_vox * n_gru_vox * n_gru_vox * filters, namew, nameb)
    fc_output = relu_layer(fc_output)
    fc_output = tf.reshape(fc_output, out_shape)
    h_tn = fc_output + slim.conv3d(h_t, filters, [3, 3, 3])
    return h_tn


"""
    Decoder
"""
def unpooling_layer(x):
    shape = x.get_shape().as_list()
    dim = len(shape[1:-1])
    out = (tf.reshape(x, [-1] + shape[-dim:]))
    for i in range(dim, 0, -1):
        out = tf.concat([out, tf.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in shape[1:-1]] + [shape[-1]]
    out = tf.reshape(out, out_size)
    return out
