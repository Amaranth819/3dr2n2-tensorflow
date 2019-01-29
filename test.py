#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:46:07 2019

@author: xulin
"""

import sys
sys.path.append('./lib')

import voxel
import dataset
import config

import tensorflow as tf
import numpy as np

def write_prediction(e, p, bs, pred_size, threshold):
    p = np.array(p)[0]
    for i in range(bs):
        filename = "./generate_models/%d_%d.binvox" % (e, i)
        voxel.write_binvox_file(p[i, ..., 1] >= threshold, filename)
        print('Write the file ' + filename + ' successfully!')

def test(graph_path, ckpt_path, input_name, gt_name, pred_name, test_category, test_size):
    bs = config.batch_size
    seq = config.sequence_length
    pred_size = config.prediction_size
    threshold = config.threshold
    cate_dataset = dataset.create_subcate_dataset(config.dataset_img_path + config.dataset_categories[test_category],
                                                  config.dataset_model_path + config.dataset_categories[test_category],
                                                  bs, config.dataset_scale, False, 1)
    iterator = cate_dataset.make_one_shot_iterator()
    batch_tensor = iterator.get_next()
    
    print("Create the testing dataset successfully!")
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(graph_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name(input_name)
        gt = graph.get_tensor_by_name(gt_name)
        pred = graph.get_tensor_by_name(pred_name)
        IoU = graph.get_tensor_by_name("IoU:0")
        loss = graph.get_tensor_by_name("loss_function/cross_entropy_loss:0")
        
        # Write model
        for i in range(test_size):
            batch_tensor = iterator.get_next()
            batch = sess.run(batch_tensor)
            print(batch[1])
            img_matrix = dataset.img2matrix(batch[0], seq)
            model_matrix = dataset.modelpath2matrix(batch[1])
            p = sess.run([pred], feed_dict = {x : img_matrix, gt : model_matrix})
            write_prediction(i, p, bs, pred_size, threshold)
        
#        # Calculate average IoU and loss
#        total_iou = []
#        total_loss = []
#        try:
#            i = 1
#            while True:
#                batch = sess.run(batch_tensor)
#                if batch[0].shape[0] is not bs:
#                    break
#                img_matrix = dataset.img2matrix(batch[0], seq)
#                model_matrix = dataset.modelpath2matrix(batch[1])
#                
#                batch_iou, batch_loss = sess.run([IoU, loss], feed_dict = {x : img_matrix, gt : model_matrix})
#                total_iou.append(batch_iou)
#                total_loss.append(batch_loss)
#                i += 1
#                if i % 20 == 0:
#                    print("After %d batches: Average IoU:%.6f, loss:%.6f." % (i, np.mean(total_iou), np.mean(total_loss)))
#        except tf.errors.OutOfRangeError:
#            print("tf.errors.OutOfRangeError!")
#        print("After testing the subcategory %d, the average IoU is %.6f and loss is %.6f." % (test_category, np.mean(total_iou), np.mean(total_loss)))
            
                
test(graph_path = config.save_model_path + config.model_name + '-100.meta',
     ckpt_path = config.save_model_path,
     input_name = "input:0",
     gt_name = "ground_truth:0",
     pred_name = "prediction:0",
     test_category = config.subcate,
     test_size = 1)