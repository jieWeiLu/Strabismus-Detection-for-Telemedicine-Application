
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
import pylab
import xml.dom.minidom

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from math import ceil,floor

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


get_ipython().run_line_magic('matplotlib', 'inline')

from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_CKPT = './frozen_inference_graph.pb'

PATH_TO_LABELS = './pascal_label_map.pbtxt'

NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():

    with tf.Session() as sess:
      
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
       
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

     
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


#读取xml文件，返回所有bndbox的[xmin,ymin,xmax,ymax]
def read_xml(path):
    a = xml.dom.minidom.parse(xml_file_path)
    root = a.documentElement
    
    object_units = root.getElementsByTagName('object')
    object_len = len(object_units)
    bndbox_data = np.zeros((object_len,4),dtype=np.int)
    
    for j,object_unit in enumerate(object_units):

        bndbox = object_unit.getElementsByTagName('bndbox')[0]
        
        xmin = bndbox.getElementsByTagName('xmin')[0]
        xmin_data = xmin.childNodes[0].data
        bndbox_data[j,0]=xmin_data
        
        ymin = bndbox.getElementsByTagName('ymin')[0]
        ymin_data = ymin.childNodes[0].data
        bndbox_data[j,1]=ymin_data
    
        xmax = bndbox.getElementsByTagName('xmax')[0]
        xmax_data = xmax.childNodes[0].data
        bndbox_data[j,2]=xmax_data
        
        ymax = bndbox.getElementsByTagName('ymax')[0]
        ymax_data = ymax.childNodes[0].data
        bndbox_data[j,3]=ymax_data
    return bndbox_data

#计算iou
def boxoverlap(a,b):    
    x1 = max(a[0],b[0])
    y1 = max(a[1],b[1])
    x2 = min(a[2],b[2])
    y2 = min(a[3],b[3])
    
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    
    inter = w * h
    aarea = (a[2]-a[0]+1) * (a[3]-a[1]+1)
    barea = (b[2]-b[0]+1) * (b[3]-b[1]+1)
    ratio = inter/(aarea+barea-inter)
    if w<=0 or h<=0:
        ratio = 0
    return ratio

#画bndbox=[xmin,ymin,xmax,ymax]
def draw_bndbox(img,box,width,color):
    draw = ImageDraw.Draw(img)
    draw.line([(box[0],box[1]),
               (box[2],box[1]),
               (box[2],box[3]),
               (box[0],box[3]),
               (box[0],box[1])], width=width, fill=color)


XML_PATH = './XML_FILE/unnormal/'
IMG_PATH = './IMG_FILE/unnormal/'
crop_path = './detection_result/crop_image/normal'
result_path = './detection_result/result_image/normal'

XML_FILE = os.listdir(XML_PATH)
XML_FILE.sort()
IMG_FILE = os.listdir(IMG_PATH)
IMG_FILE.sort()

IMAGE_SIZE = (12, 8)

iou_all_unnormal=[]
unnormal_name=[]
for i,image_path in enumerate(IMG_FILE):

  image_type=image_path[-3:]
  if image_type=='png':
      continue
  
  xml_file_path = os.path.join(XML_PATH,XML_FILE[i])
  img_file_path = os.path.join(IMG_PATH,IMG_FILE[i])
  raw_image_name = os.path.basename(img_file_path)
  
  bndbox_datas = read_xml(xml_file_path)

  image = Image.open(img_file_path)
  image_size = np.array([image.size[0],image.size[1],image.size[0],image.size[1]])

  image_np = load_image_into_numpy_array(image)

  output_dict = run_inference_for_single_image(image_np, detection_graph)

  detection_boxes = output_dict.get('detection_boxes')
  detection_scores = output_dict.get('detection_scores')

  output_boxs = []
  for index,val in enumerate(detection_scores):
      if index == 0:
          box = np.array([detection_boxes[index,1],detection_boxes[index,0],detection_boxes[index,3],detection_boxes[index,2]])
          output_box = image_size * box
          output_box = np.array([ceil(output_box[0]),ceil(output_box[1]),floor(output_box[2]),floor(output_box[3])])
          output_boxs.append(output_box)
          cropped_image = image.crop(output_box)
          crop_image_path = os.path.join(crop_path,raw_image_name)
          cropped_image.save(crop_image_path)
  output_boxs = np.array(output_boxs) 
  
  if len(output_boxs):
      max_ious=[]
      for m in range(output_boxs.shape[0]):
          ious=[]
          for k in range(bndbox_datas.shape[0]):
              output_box = output_boxs[m,:]
              bndbox_data = bndbox_datas[k,:]
              draw_bndbox(image,output_box,1,'red')
              iou = boxoverlap(bndbox_data,output_box)
              ious.append(iou)
          max_iou = max(ious)
          max_ious.append(max_iou)
      iou_all_unnormal.append(max(max_ious))
      print(max(max_ious))
  else:
      max_ious=[]
      max_iou = 0
      max_ious.append(max_iou)
      iou_all_unnormal.append(max(max_ious))
      print(max(max_ious))

  for k in range(bndbox_datas.shape[0]):
      bndbox_data = bndbox_datas[k,:]
      draw_bndbox(image,bndbox_data,1,'black')
  
  unnormal_name.append(image_path)
    
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image)
  
  result_image_path = os.path.join(result_path,raw_image_name)
  plt.savefig(result_image_path,bbox_inches='tight')
  plt.close('all')
  print('unnormal:',i)
  

  
XML_PATH = './XML_FILE/normal/'
IMG_PATH = './IMG_FILE/normal/'
crop_path = './detection_result/crop_image/normal'
result_path = './detection_result/result_image/normal'

XML_FILE = os.listdir(XML_PATH)
XML_FILE.sort()
IMG_FILE = os.listdir(IMG_PATH)
IMG_FILE.sort()

IMAGE_SIZE = (12, 8)

iou_all_normal=[]
normal_name=[]
for i,image_path in enumerate(IMG_FILE):

  image_type=image_path[-3:]
  if image_type=='png':
      continue
  
  xml_file_path = os.path.join(XML_PATH,XML_FILE[i])
  img_file_path = os.path.join(IMG_PATH,IMG_FILE[i])
  raw_image_name = os.path.basename(img_file_path)
  
  bndbox_datas = read_xml(xml_file_path)

  image = Image.open(img_file_path)
  image_size = np.array([image.size[0],image.size[1],image.size[0],image.size[1]])

  image_np = load_image_into_numpy_array(image)

  output_dict = run_inference_for_single_image(image_np, detection_graph)

  detection_boxes = output_dict.get('detection_boxes')
  detection_scores = output_dict.get('detection_scores')

  output_boxs = []
  for index,val in enumerate(detection_scores):
      if index == 0:
          box = np.array([detection_boxes[index,1],detection_boxes[index,0],detection_boxes[index,3],detection_boxes[index,2]])
          output_box = image_size * box
          output_box = np.array([ceil(output_box[0]),ceil(output_box[1]),floor(output_box[2]),floor(output_box[3])])
          output_boxs.append(output_box)
          cropped_image = image.crop(output_box)
          crop_image_path = os.path.join(crop_path,raw_image_name)
          cropped_image.save(crop_image_path)
  output_boxs = np.array(output_boxs) 
  
  if len(output_boxs):
      max_ious=[]
      for m in range(output_boxs.shape[0]):
          ious=[]
          for k in range(bndbox_datas.shape[0]):
              output_box = output_boxs[m,:]
              bndbox_data = bndbox_datas[k,:]
              draw_bndbox(image,output_box,1,'red')
              iou = boxoverlap(bndbox_data,output_box)
              ious.append(iou)
          max_iou = max(ious)
          max_ious.append(max_iou)
      iou_all_normal.append(max(max_ious))
      print(max(max_ious))
  else:
      max_ious=[]
      max_iou = 0
      max_ious.append(max_iou)
      iou_all_normal.append(max(max_ious))
      print(max(max_ious))

  for k in range(bndbox_datas.shape[0]):
      bndbox_data = bndbox_datas[k,:]
      draw_bndbox(image,bndbox_data,1,'black')

  normal_name.append(image_path)
  
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image)
  
  result_image_path = os.path.join(result_path,raw_image_name)
  plt.savefig(result_image_path,bbox_inches='tight')
  plt.close('all')
  print('normal:',i)


output_file='./'
output_path = os.path.join(output_file,'iou_result.npy')    
np.save(output_path,np.asarray([iou_all_unnormal,unnormal_name,iou_all_normal,normal_name]))  
