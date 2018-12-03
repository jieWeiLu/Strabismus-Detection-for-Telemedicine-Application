#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 19:52:08 2018

@author: JinganFeng
"""

import numpy as np

result = np.load('./detection_result/iou_result.npy')
iou_unnormal = result[0]
iou_unnormal_name = result[1]
iou_normal = result[2]
iou_normal_name = result[3]

sum1=sum(iou_unnormal)
sum2=sum(iou_normal)
len1=len(iou_unnormal)
len2=len(iou_normal)
avr=(sum1+sum2)/(len1+len2)
print("iou_mean:",avr)
