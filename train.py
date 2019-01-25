#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:12:26 2019

@author: xulin
"""
import sys
sys.path.append('./network')
sys.path.append('./lib')

import res_gru_net
import dataset, config

import tensorflow as tf

def evaluate_batch_prediction(pred, gt, threshold):
    pred_occupy = pred[..., 1] >= threshold
#    I1 = np.sum(np.logical_and(pred_occupy, gt))
#    I2 = np.sum(np.logical_or(pred_occupy, gt))
    
    I1 = tf.reduce_sum(tf.cast(tf.math.logical_and(pred_occupy, tf.cast(gt, tf.bool)), tf.float32))
    I2 = tf.reduce_sum(tf.cast(tf.math.logical_or(pred_occupy, tf.cast(gt, tf.bool)), tf.float32))
    IoU = tf.math.divide(I1, I2, name = "IoU")
    
    tf.summary.scalar("IoU", IoU)
    return IoU

def softmax_cross_entropy(pred, gt):
    # pred: [bs, 32, 32, 32, 2]
    # gt: [bs, 32, 32, 32]
    with tf.name_scope("loss_function"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt, logits = pred), name = "cross_entropy_loss")
        tf.summary.scalar("loss", loss)
        return loss

class Train(object):
    def __init__(self):
        self.input = tf.placeholder(dtype = tf.float32, shape = config.input_size, name = "input")
        self.ground_truth = tf.placeholder(dtype = tf.int32, shape = config.ground_truth_size, name = "ground_truth")
        
    def train(self, subcate):  
        # tf.reset_default_graph()
        sess = tf.Session()
        
        pred = res_gru_net.build_network(self.input)
        loss = softmax_cross_entropy(pred, self.ground_truth)
        global_steps = tf.Variable(0, trainable = False, name = "global_step")
        learing_rate = tf.train.exponential_decay(config.init_learning_rate, global_steps, config.lr_decay_steps, config.lr_decay, staircase = True, name = "learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate = learing_rate, name = "adam_optimizer").minimize(loss, global_step = global_steps, name = "adam_minimizer")
        IoU = evaluate_batch_prediction(pred, self.ground_truth, config.threshold)
        
        sess.run(tf.global_variables_initializer())
        # prediction:0
        print pred
        
        train_writer = tf.summary.FileWriter(config.log_path, sess.graph)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        
        category = config.dataset_categories[subcate]
        #np.random.shuffle(all_categories)
        #big_dataset = dataset.create_big_dataset(config.dataset_img_path, config.dataset_model_path, all_categories, config.batch_size, config.dataset_scale, True)
        #iterator = big_dataset.make_one_shot_iterator()
        
        big_dataset = dataset.create_subcate_dataset(config.dataset_img_path + category,
                                                     config.dataset_model_path + category,
                                                     config.batch_size,
                                                     config.dataset_scale,
                                                     True)
        iterator = big_dataset.make_one_shot_iterator()
        batch_tensor = iterator.get_next()
        print("Create the training dataset successfully!")
        
        # sess.graph.finalize()
        
        for e in range(config.epoch):
            batch = sess.run(batch_tensor)
            
            img_matrix = dataset.img2matrix(batch[0], config.sequence_length)
            model_matrix = dataset.modelpath2matrix(batch[1])
            
            summary, gs, l, iou, lr, _ = sess.run([merged, global_steps, loss, IoU, learing_rate, optimizer], feed_dict = {self.input : img_matrix, self.ground_truth : model_matrix})
            train_writer.add_summary(summary, gs)
            
            print("Global steps %d -- loss:%.6f, IoU:%.6f, lr:%.6f" % (gs, l, iou, lr))
            
            if gs % config.save_model_step == 0:
                saver.save(sess, config.save_model_path + config.model_name, global_step = config.save_model_step)
                print("Save the model successfully!")
                
        train_writer.close()
        
t = Train()
t.train(config.subcate)