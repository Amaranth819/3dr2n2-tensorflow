#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:14:27 2019

@author: xulin
"""

"""
    Dataset paths
"""
dataset_img_path = '../../github/3D-R2N2/ShapeNet/ShapeNetRendering/'
dataset_model_path = '../../github/3D-R2N2/ShapeNet/ShapeNetVox32/'
dataset_categories = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243',  
                     '04401088', '04530566']

subcategory_size = 24
dataset_scale = (0.8, 0.2) # training_dataset_scale : testing_dataset_scale = 4 : 1


"""
    Training parameters
"""
init_learning_rate = 1e-4
lr_decay = 0.95
lr_decay_steps = 100
epoch = 200

subcate = 0
save_model_step = 100
save_model_path = './model/res_gru_' + '%d' % (subcate) + '/'
model_name = 'res_gru_' + '%d' % (subcate)

log_path = './log/train/' + '%d' % (subcate)


"""
    Network parameters
"""
batch_size = 4
sequence_length = 5
img_size = [127, 127, 3]
input_size = [batch_size, sequence_length] + img_size

model_size = [32, 32, 32]
ground_truth_size = [batch_size] + model_size

prediction_size = [batch_size] + model_size + [2]

threshold = 0.4