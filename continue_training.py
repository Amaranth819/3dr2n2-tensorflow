#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 01:08:40 2019

@author: xulin
"""
import sys
sys.path.append('./lib')

import dataset
import config

import tensorflow as tf

def continue_training(model_path, graph_path, ckpt_path, input_name, gt_name, epoch, subcate, bs):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(graph_path)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    
    train_writer = tf.summary.FileWriter(config.log_path, sess.graph)
    merged = tf.summary.merge_all()
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name(input_name)
    gt = graph.get_tensor_by_name(gt_name)
    loss = graph.get_tensor_by_name("loss_function/cross_entropy_loss:0")
    IoU = graph.get_tensor_by_name("IoU:0")
    global_steps = graph.get_tensor_by_name("global_step:0")
    learning_rate = graph.get_tensor_by_name("learning_rate:0")
    optimizer = graph.get_tensor_by_name("adam_minimizer:0")
    
    category = config.dataset_categories[subcate]
    big_dataset = dataset.create_subcate_dataset(config.dataset_img_path + category,
                                                 config.dataset_model_path + category,
                                                 config.batch_size,
                                                 config.dataset_scale,
                                                 True)
    iterator = big_dataset.make_one_shot_iterator()
    batch_tensor = iterator.get_next()
    print("Create the training dataset successfully!")
    
    e = 0
    while e < epoch:
        batch = sess.run(batch_tensor)
        if batch[0].shape[0] is not bs:
            continue
        
        img_matrix = dataset.img2matrix(batch[0], config.sequence_length)
        model_matrix = dataset.modelpath2matrix(batch[1])
        
        summary, l, iou, gs, lr, _ = sess.run([merged, loss, IoU, global_steps, learning_rate, optimizer], feed_dict = {x : img_matrix, gt : model_matrix})
        train_writer.add_summary(summary, gs)
        
        print("Global steps %d -- loss:%.6f, IoU:%.6f, lr:%.6f" % (gs, l, iou, lr))
        
        if gs % config.save_model_step == 0:
            saver.save(sess, model_path, global_step = config.save_model_step)
            print("Save the model successfully!")
            
        e += 1
            
    train_writer.close()
    
continue_training(
        model_path = config.save_model_path + config.model_name,
        graph_path = config.save_model_path + config.model_name + '-' + '%d' % (config.save_model_step) + '.meta',
        ckpt_path = config.save_model_path,
        input_name = "input:0",
        gt_name = "ground_truth:0",
        epoch = 400,
        subcate = config.subcate,
        bs = config.batch_size)