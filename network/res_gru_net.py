#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:24:27 2019

@author: xulin
"""
import sys
sys.path.append("../lib")

import layers
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

fc_layer_size = [1024]
n_deconv_filters = [128, 128, 128, 64, 32, 1]
n_gru_vox = 4

"""
    Encoder
"""
# Build the encoder network
def build_encoder(x):
    # The encoder uses the deep residual network.
    outputs = []
    pooling = [1, 2, 2, 1]
    
    shape = x.get_shape().as_list()
    bs = shape[0]
    seq = shape[1]
    temp_shape = [bs * seq] + shape[2:]
    x = tf.reshape(x, temp_shape)
    # print x.get_shape().as_list()
    
    # layer 0
    with tf.variable_scope("encoder_layer0", reuse = tf.AUTO_REUSE):
        conv0_0 = layers.conv_layer(name = "conv0_0", x = x, filter_shape = layers.create_variable("filter0_0", shape = [7, 7, 3, 96]))
        conv0_0 = layers.batch_normalization(conv0_0)
        conv0_0 = layers.relu_layer(conv0_0)
        conv0_1 = layers.conv_layer(name = "conv0_1", x = conv0_0, filter_shape = layers.create_variable("filter0_1", shape = [3, 3, 96, 96]))
        conv0_1 = layers.batch_normalization(conv0_1)
        conv0_1 = layers.relu_layer(conv0_1)
        shortcut0 = layers.conv_layer(name = "shortcut", x = x, filter_shape = layers.create_variable("filter0_2", shape = [1, 1, 3, 96]))
        shortcut0 = layers.batch_normalization(shortcut0)
        shortcut0 = layers.relu_layer(shortcut0)
        layer0 = layers.pooling_layer("pooling", conv0_1 + shortcut0, pooling)
        outputs.append(layer0) # [bs * size, 64, 64, 96]
        
    # layer 1
    with tf.variable_scope("encoder_layer1", reuse = tf.AUTO_REUSE):
        conv1_0 = layers.conv_layer(name = "conv1_0", x = layer0, filter_shape = layers.create_variable("filter1_0", shape = [3, 3, 96, 128]))
        conv1_0 = layers.batch_normalization(conv1_0)
        conv1_0 = layers.relu_layer(conv1_0)
        conv1_1 = layers.conv_layer(name = "conv1_1", x = conv1_0, filter_shape = layers.create_variable("filter1_1", shape = [3, 3, 128, 128]))
        conv1_1 = layers.batch_normalization(conv1_1)
        conv1_1 = layers.relu_layer(conv1_1)
        shortcut1 = layers.conv_layer(name = "shortcut", x = layer0, filter_shape = layers.create_variable("filter1_2", shape = [1, 1, 96, 128]))
        shortcut1 = layers.batch_normalization(shortcut1)
        shortcut1 = layers.relu_layer(shortcut1)
        layer1 = layers.pooling_layer("pooling", conv1_1 + shortcut1, pooling)
        outputs.append(layer1) # [bs * size, 32, 32, 128]
        
    # layer 2
    with tf.variable_scope("encoder_layer2", reuse = tf.AUTO_REUSE):
        conv2_0 = layers.conv_layer(name = "conv2_0", x = layer1, filter_shape = layers.create_variable("filter2_0", shape = [3, 3, 128, 256]))
        conv2_0 = layers.batch_normalization(conv2_0)
        conv2_0 = layers.relu_layer(conv2_0)
        conv2_1 = layers.conv_layer(name = "conv2_1", x = conv2_0, filter_shape = layers.create_variable("filter2_1", shape = [3, 3, 256, 256]))
        conv2_1 = layers.batch_normalization(conv2_1)
        conv2_1 = layers.relu_layer(conv2_1)
        shortcut2 = layers.conv_layer(name = "shortcut", x = layer1, filter_shape = layers.create_variable("filter2_2", shape = [1, 1, 128, 256]))
        shortcut2 = layers.batch_normalization(shortcut2)
        shortcut2 = layers.relu_layer(shortcut2)
        layer2 = layers.pooling_layer("pooling", conv2_1 + shortcut2, pooling)
        outputs.append(layer2) # [bs * size, 16, 16, 256]
        
    # layer 3
    with tf.variable_scope("encoder_layer3", reuse = tf.AUTO_REUSE):
        conv3_0 = layers.conv_layer(name = "conv3_0", x = layer2, filter_shape = layers.create_variable("filter3_0", shape = [3, 3, 256, 256]))
        conv3_0 = layers.batch_normalization(conv3_0)
        conv3_0 = layers.relu_layer(conv3_0)
        conv3_1 = layers.conv_layer(name = "conv3_1", x = conv3_0, filter_shape = layers.create_variable("filter3_1", shape = [3, 3, 256, 256]))
        conv3_1 = layers.batch_normalization(conv3_1)
        conv3_1 = layers.relu_layer(conv3_1)
        layer3 = layers.pooling_layer("pooling", conv3_1, pooling)
        outputs.append(layer3) # [bs * size, 8, 8, 256]
        
    # layer 4
    with tf.variable_scope("encoder_layer4", reuse = tf.AUTO_REUSE):
        conv4_0 = layers.conv_layer(name = "conv4_0", x = layer3, filter_shape = layers.create_variable("filter4_0", shape = [3, 3, 256, 256]))
        conv4_0 = layers.batch_normalization(conv4_0)
        conv4_0 = layers.relu_layer(conv4_0)
        conv4_1 = layers.conv_layer(name = "conv4_1", x = conv4_0, filter_shape = layers.create_variable("filter4_1", shape = [3, 3, 256, 256]))
        conv4_1 = layers.batch_normalization(conv4_1)
        conv4_1 = layers.relu_layer(conv4_1)
        shortcut4 = layers.conv_layer(name = "shortcut", x = layer3, filter_shape = layers.create_variable("filter4_2", shape = [1, 1, 256, 256]))
        shortcut4 = layers.batch_normalization(shortcut4)
        shortcut4 = layers.relu_layer(shortcut4)
        layer4 = layers.pooling_layer("pooling", conv4_1 + shortcut4, pooling)
        outputs.append(layer4) # [bs * size, 4, 4, 256]
        
    # layer 5
    with tf.variable_scope("encoder_layer5", reuse = tf.AUTO_REUSE):
        conv5_0 = layers.conv_layer(name = "conv5_0", x = layer4, filter_shape = layers.create_variable("filter5_0", shape = [3, 3, 256, 256]))
        conv5_0 = layers.batch_normalization(conv5_0)
        conv5_0 = layers.relu_layer(conv5_0)
        conv5_1 = layers.conv_layer(name = "conv5_1", x = conv5_0, filter_shape = layers.create_variable("filter5_1", shape = [3, 3, 256, 256]))
        conv5_1 = layers.batch_normalization(conv5_1)
        conv5_1 = layers.relu_layer(conv5_1)
        shortcut5 = layers.conv_layer(name = "shortcut", x = layer4, filter_shape = layers.create_variable("filter5_2", shape = [1, 1, 256, 256]))
        shortcut5 = layers.batch_normalization(shortcut5)
        shortcut5 = layers.relu_layer(shortcut5)
        layer5 = layers.pooling_layer("pooling", conv5_1 + shortcut5, pooling)
        outputs.append(layer5) # [bs * size, 2, 2, 256]
    
    final_shape = [bs, seq, fc_layer_size[0]]
    # Flatten layer and fully connected layer
    flatten = layers.flatten_layer(layer5)
    outputs.append(flatten)
    
    with tf.variable_scope("fc_layer", reuse=tf.AUTO_REUSE):
        layer_fc = layers.fully_connected_layer(flatten, fc_layer_size[0], "fclayer_w", "fclayer_b")
        # layer_fc = layers.batch_normalization(layer_fc)
        layer_fc = layers.relu_layer(layer_fc)
        outputs.append(layer_fc) # [bs * size, 1024]
    
    # [bs, size, 1024]
    return tf.reshape(outputs[-1], final_shape)

def test_encoder():
    y = tf.constant(np.ones((16, 5, 127, 127, 3), dtype = np.float32))
    output = build_encoder(y)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print("OK!")
    print output

# test_encoder()

"""
    GRU
"""
# Calculate the output of a recurrence
def recurrence(h_t, feature_x, filters, n_gru_vox, index):
    u_t = tf.sigmoid(layers.fcconv3d_layer(h_t, feature_x, filters, n_gru_vox, "u_%d_weight" % (index), "u_%d_bias" % (index)))
    r_t = tf.sigmoid(layers.fcconv3d_layer(h_t, feature_x, filters, n_gru_vox, "r_%d_weight" % (index), "r_%d_bias" % (index)))
    h_tn = (1.0 - u_t) * h_t + u_t * tf.tanh(layers.fcconv3d_layer(r_t * h_t, feature_x, filters, n_gru_vox, "h_%d_weight" % (index), "h_%d_bias" % (index)))
    return h_tn

# Build 3d gru network
def build_3dgru(features):
    # "features" is a tensor of size [bs, size, 1024]
    # Split the "features" into many sequences.    
    with tf.variable_scope("gru", reuse = tf.AUTO_REUSE):
        shape = features.get_shape().as_list()
#        newshape = [shape[1], shape[0], shape[2]] # [size, bs, 1024]
#        features = tf.reshape(features, newshape)
        h = [None for _ in range(shape[1] + 1)]
        h[0] = tf.zeros(shape = [shape[0], n_gru_vox, n_gru_vox, n_gru_vox, n_deconv_filters[0]], dtype = tf.float32)
        for i in range(shape[1]):
            fc = features[:, i, ...]
            h[i + 1] = recurrence(h[i], fc, n_deconv_filters[0], n_gru_vox, i)
        # [bs, 4, 4, 4, 128]
        return h[-1]
    
def test_3dgru():
    x = np.ones((16, 5, 1024), dtype = np.float32)
    y = tf.constant(x)
    output = build_3dgru(y)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print("OK!")
    print output


"""
    Decoder
"""
def build_decoder(h_f):
    with tf.variable_scope("decoder_layer0", reuse = tf.AUTO_REUSE):
        unpool0 = layers.unpooling_layer(h_f)
        conv0_0 = slim.conv3d(unpool0, n_deconv_filters[1], 3)
        conv0_1 = slim.conv3d(conv0_0, n_deconv_filters[1], 3)
        output0 = unpool0 + conv0_1
        
    with tf.variable_scope("decoder_layer1", reuse = tf.AUTO_REUSE):
        unpool1 = layers.unpooling_layer(output0)
        conv1_0 = slim.conv3d(unpool1, n_deconv_filters[2], 3)
        conv1_1 = slim.conv3d(conv1_0, n_deconv_filters[2], 3)
        output1 = unpool1 + conv1_1
        
    with tf.variable_scope("decoder_layer2", reuse = tf.AUTO_REUSE):
        unpool2 = layers.unpooling_layer(output1)
        shortcut2 = slim.conv3d(unpool2, n_deconv_filters[3], 1)
        conv2_0 = slim.conv3d(unpool2, n_deconv_filters[3], 3)
        conv2_1 = slim.conv3d(conv2_0, n_deconv_filters[3], 3)
        output2 = shortcut2 + conv2_1
        
    with tf.variable_scope("decoder_layer3", reuse = tf.AUTO_REUSE):
        conv3_0 = slim.conv3d(output2, n_deconv_filters[4], 3)
        conv3_1 = slim.conv3d(conv3_0, n_deconv_filters[4], 3)
        conv3_2 = slim.conv3d(conv3_0, n_deconv_filters[4], 3)
        output3 = conv3_1 + conv3_2
        
    with tf.variable_scope("decoder_layer4", reuse = tf.AUTO_REUSE):
        output4 = slim.conv3d(output3, n_deconv_filters[5], 3)
    
    # [bs, 32, 32, 32, 2]
    # mask
    prediction = tf.concat([output4, 1.0 - output4], -1, name = "prediction")
    return prediction

def test_decoder():
    h = tf.constant(np.random.random((16, 4, 4, 4, 128)))
    out = build_decoder(h)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    print out.get_shape()
    print("OK!")
    

"""
    Res_gru_network
"""
def build_network(x):
    encoder_output = build_encoder(x)
    print encoder_output.get_shape().as_list()
    gru_output = build_3dgru(encoder_output)
    print gru_output.get_shape().as_list()
    pred = build_decoder(gru_output)
    print pred.get_shape().as_list()
    return pred

def test_network():
    y = tf.constant(np.random.random((16, 5, 127, 127, 3)))
    pred = build_network(y)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    print("OK!")
    # name: prediction:0
    print pred

# test_network()