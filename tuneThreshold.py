#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    fnr = fnr*100 # 错误拒绝�?    
    fpr = fpr*100 # 错误接收�?    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = min(fpr[idxE],fnr[idxE]) # 等错误率

    minDCF = numpy.min((0.01*fnr + 0.99*fpr)) # minDCF
    
    return (eer, minDCF);
