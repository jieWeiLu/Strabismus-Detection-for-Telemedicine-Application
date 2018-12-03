#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 19:52:08 2018

@author: JinganFeng
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc

result = np.load('./network_result/rfcn_test.npy')

#label: {unnormal:1, normal:0}

real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

y=real_results1
scores = probabilitys1
fpr,tpr,t1hresholds = metrics.roc_curve(y,scores)
plt.plot(fpr,tpr,marker='o')
plt.show()
AUC = auc(fpr,tpr)
print("auc:",AUC)

TP=0
FP=0
FN=0
TN=0

for i in range(len(real_results1)):
    if real_results1[i]==1 and dect_results1[i]==1:
        TP += 1
    if real_results1[i]==0 and dect_results1[i]==1:
        FP += 1
    if real_results1[i]==1 and dect_results1[i]==0:
        FN += 1
    if real_results1[i]==0 and dect_results1[i]==0:
        TN += 1
print("TP:",TP)
print("TN:",TN)
print("FP:",FP)
print("FN:",FN)

TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("specificity:",TNR)
print("sensitive:",TPR)
print("accuracy:",ACC)
