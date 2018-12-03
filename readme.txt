



The code and the algorithm are for non-commercial use only.

Paper : "Automated Strabismus Detection for Telemedicine Applications"

Author: Jiewei Lu, Zhun Fan, Ce Zheng, Jingan Feng, Longtao Huang, Wenji Li, Erik D. Goodman

        (12jwlu1@stu.edu.cn, zfan@stu.edu.cn, zhengce@hotmail.com, 13jafeng@stu.edu.cn, 17lthuang@stu.edu.cn, , liwj@stu.edu.cn, goodman@egr.msu.edu)

Date  : December 2, 2018

Version : 2.0

Copyright (c) 2018, Jiewei Lu, Jingan Feng.

--------------------------------------------------------------

Notes:
  1) Tensorflow, an open source machine learning framework, is required for the implementation.

  2) R-FCN for eye region segmentation, is based on the TensorFlow Object Detection API. Please refer to the source code at https://github.com/tensorflow/models/tree/master/research/object_detection

  3) Deep CNN for eye region classification, is based on the TensorFlow-Slim image classification model library. Please refer to the source code at https://github.com/tensorflow/models/tree/master/research/slim

  4) scikit-learn, a machine learning library, is required for the evaluation metrics (ROC & AUC).

  These libraries can be easily set by packet manager on linux systems

--------------------------------------------------------------

This folder contains two sub-directories:

  - Eye_Region_Segmentation
     - eye_detection.py       the source code of using R-FCN to segment eye region
     - iou.py                 the source code of displaying the mean IOU result of Segmentation
     - IMG_FILE               contains some example images for testing R-FCN
     - XML_FILE               contains corresponding bounding box information for each example image
     - detection_result       contains the numpy file saving IOU output of R-FCN for each images in test set

  - Strabismus_Diagnosis
     - eye_classification.py  the source code of using deep CNN to classify eye region
     - roc_auc.py             the source code of calculating and displaying evaluation metrics of our deep CNN
     - CROP_IMAGE             contains some example images for testing the deep CNNs
     - network                contains the design file of the network architecture
     - network_result        contains the numpy file saving goundtruth labels and detected results of test set


