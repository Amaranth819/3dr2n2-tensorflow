#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:16:00 2019

@author: xulin
"""

import tensorflow as tf

def decode_img(path_tensor, img_size, channels = 3):
    tf.random.set_random_seed(2333)
    img = tf.read_file(path_tensor)
    img = tf.image.decode_image(img, channels)
    img = tf.image.random_crop(img, img_size)
    return img

def data_augmentation():
    pass