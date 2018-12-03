#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:54:37 2018

@author: JinganFeng
"""
import os
import tensorflow as tf
import numpy as np
import gc

import pylab
from matplotlib import pyplot as plt

from datasets import imagenet
from nets import inception
from nets import alexnet
from preprocessing import my_preprocessing

from tensorflow.contrib import slim

output_file = './'
output_path = os.path.join(output_file,'rfcn_test.npy')
output_path_fc7 = os.path.join(output_file,'rfcn_test_fc7.npy')

real_results_group=[]
dect_results_group=[]
probabilitys_group=[]
names_group=[]
fc7s_group=[]

checkpoints_dir = './network1/model.ckpt-4832'

path_unnormal = '/home/JinganFeng/Downloads/straPack/eye_2/detection_result/crop_image/unnormal/'
path_normal = '/home/JinganFeng/Downloads/straPack/eye_2/detection_result/crop_image/normal/'

image_names = os.listdir(path_unnormal)
image_names.sort()

tf.reset_default_graph()

images = tf.placeholder(tf.float32, [1,224,224,3], name='input_images')

with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):    
    logits, end_points1 = alexnet.alexnet_v2(images, num_classes=2, is_training=False)
probabilities1 = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(
    checkpoints_dir,
    slim.get_variables_to_restore(),ignore_missing_vars=True)
    
sess = tf.Session()
init_fn(sess)

for j,image_name in enumerate(image_names):

    print('unnormal:',j)
    
    image_path = os.path.join(path_unnormal,image_name)
    image_string=tf.gfile.FastGFile(image_path,'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3) 
    
    image_size = alexnet.alexnet_v2.default_image_size    
    processed_image = my_preprocessing.preprocess_image(image, image_size, image_size,None,None)
    processed_images  = tf.expand_dims(processed_image, 0)

    np_image=sess.run(processed_images)
    
    probabilities, end_points = sess.run([probabilities1, end_points1],feed_dict={images:np_image})
    
    fc7_end_points = end_points['alexnet_v2/fc7']
    fc7_end_points_reshape = fc7_end_points.reshape(512)
    
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    
    names = []
    names.append('normal_crop')
    names.append('unnormal_crop')
    
    for i in range(2):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

    binary = sorted_inds[0]
    probability = probabilities[1]
    name = names[sorted_inds[0]]
    fc7 = fc7_end_points_reshape
    
    real_results_group.append(1)
    dect_results_group.append(binary)
    probabilitys_group.append(probability)
    names_group.append(image_name)
    
    fc7s_group.append(fc7)
    if j % 100 == 0:
        sess.close()
        gc.collect()    
        sess = tf.Session()           
        init_fn(sess)
    
    
            
image_names = os.listdir(path_normal)
image_names.sort()

tf.reset_default_graph()

images = tf.placeholder(tf.float32, [1,224,224,3], name='input_images')

with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):    
    logits, end_points1 = alexnet.alexnet_v2(images, num_classes=2, is_training=False)
probabilities1 = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(
    checkpoints_dir,
    slim.get_variables_to_restore(),ignore_missing_vars=True)
    
sess = tf.Session()
init_fn(sess)

for j,image_name in enumerate(image_names):

    print('normal:',j)
    
    image_path = os.path.join(path_normal,image_name)
    image_string=tf.gfile.FastGFile(image_path,'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    image_size = alexnet.alexnet_v2.default_image_size    
    processed_image = my_preprocessing.preprocess_image(image, image_size, image_size,None,None)
    processed_images  = tf.expand_dims(processed_image, 0)

    np_image=sess.run(processed_images)
    
    probabilities, end_points = sess.run([probabilities1, end_points1],feed_dict={images:np_image})
    
    fc7_end_points = end_points['alexnet_v2/fc7']
    fc7_end_points_reshape = fc7_end_points.reshape(512)
    
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    
    names = []
    names.append('normal_crop')
    names.append('unnormal_crop')
    
    for i in range(2):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

    binary = sorted_inds[0]
    probability = probabilities[1]
    name = names[sorted_inds[0]]
    fc7 = fc7_end_points_reshape
    
    real_results_group.append(0)
    dect_results_group.append(binary)
    probabilitys_group.append(probability)
    names_group.append(image_name)
    
    fc7s_group.append(fc7)  
    
    if j % 100 == 0:
        sess.close()
        gc.collect()    
        sess = tf.Session()           
        init_fn(sess)        

np.save(output_path,np.asarray([real_results_group,dect_results_group,probabilitys_group,names_group]))
np.save(output_path_fc7,np.asarray(fc7s_group))     
