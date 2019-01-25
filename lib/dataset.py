#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:56:39 2019

@author: xulin
"""

import tensorflow as tf
import numpy as np
import glob
import data_augmentation
import voxel

"""
    File directory processing.
"""
# Get all subcategories.
def get_all_subcategory_paths(path):
    return glob.glob(path + "/*")

# Return all image paths under a subcategory.
def get_subcategory_images(subcate_path):
    return glob.glob(subcate_path + "/*/*" + "png")

# Return all model paths under a subcategory.
def get_subcategory_model(subcate_path):
    return glob.glob(subcate_path + "/*" + "binvox")

"""
    Create datasets.
"""
# Create datasets from a subcategory.
def create_subcate_dataset(img_path, model_path, bs, scale, is_train, repeat_size = None):
    category_img_paths = get_all_subcategory_paths(img_path)
    category_model_paths = get_all_subcategory_paths(model_path)
    
    # Image paths
    all_images = []
    for subcategory_img_path in category_img_paths:
        subcategory_imgs = get_subcategory_images(subcategory_img_path)
        all_images.append(subcategory_imgs)
    
    # Model paths
    all_models = []
    for subcategory_model_path in category_model_paths:
        subcategory_model = get_subcategory_model(subcategory_model_path)
        all_models.append(subcategory_model)
    
    training_size = int(len(all_models) * scale[0])
    if is_train is True:
        # Training dataset
        all_images = all_images[:training_size]
        all_models = all_models[:training_size]
    else:
        # Testing dataset
        all_images = all_images[training_size:]
        all_models = all_models[training_size:]
        
    print len(all_images)
    print len(all_models)
    
    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_models))
    dataset = dataset.map(map_batch)
    if repeat_size is None:
        dataset = dataset.shuffle(2000).batch(bs).repeat()
    else:
        dataset = dataset.shuffle(2000).batch(bs).repeat(repeat_size)
    return dataset

def create_big_dataset(img_path, model_path, categories, bs, scale, is_train, repeat_size = None):
    imgs = []
    models = []
    for cate in categories:
        category_img_paths = get_all_subcategory_paths(img_path + cate)
        category_model_paths = get_all_subcategory_paths(model_path + cate)
        
        cate_imgs = []
        for subcategory_img_path in category_img_paths:
            subcategory_imgs = get_subcategory_images(subcategory_img_path)
            cate_imgs.append(subcategory_imgs)
            
        cate_models = []
        for subcategory_model_path in category_model_paths:
            subcategory_model = get_subcategory_model(subcategory_model_path)
            cate_models.append(subcategory_model)
            
        training_size = int(len(cate_models) * scale[0])
        
        if is_train is True:
            # Training dataset
            cate_imgs = cate_imgs[:training_size]
            cate_models = cate_models[:training_size]
        else:
            # Testing dataset
            cate_imgs = cate_imgs[training_size:]
            cate_models = cate_models[training_size:]
            
        imgs = imgs + cate_imgs
        models = models + cate_models
    
    print len(imgs)
    print len(models)
    
    big_dataset = tf.data.Dataset.from_tensor_slices((imgs, models))
    big_dataset = big_dataset.map(map_batch)
    if repeat_size is None:
        big_dataset = big_dataset.shuffle(1000).batch(bs).repeat()
    else:
        big_dataset = big_dataset.shuffle(1000).batch(bs).repeat(repeat_size)
    
    print("Create the dataset successfully!")
    
    return big_dataset

def map_batch(img_batch, model_batch):
    data = []
    for each in range(img_batch.get_shape().as_list()[0]):
        img = data_augmentation.decode_img(img_batch[each], [127, 127, 3])
        data.append(img)
    data = tf.stack(data, 0)
    return data, model_batch

"""
    Feed dict
"""
def img2matrix(image_batch, sequence_length):
#    subcate_size = np.array(image_batch).shape[1]
#    start = np.random.randint(subcate_size - sequence_length + 1)
#    images = image_batch[:, start:start+sequence_length, ...]
#    return np.array(images)
    shape = list(np.array(image_batch).shape)
    start = np.random.randint(shape[1] - sequence_length + 1)
    images = np.array(image_batch[:, start:start+sequence_length, ...])
    return np.array(images)
    

def modelpath2matrix(gt_array):
    gt = []
    for paths in gt_array:
        model = voxel.read_voxel_data(paths[0])
        gt.append(model)
    gt = np.array(gt)
    return gt