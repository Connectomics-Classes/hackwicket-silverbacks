#!/usr/bin/python

#imports
import numpy as np
import cv2
import sys
from mayavi import mlab
from mayavi.mlab import *

actual = np.load('../data/actual.npy')
print actual.shape
predicted = np.load('../data/predicted.npy')
#set arrays to bools of either detected or nt
actual[actual > 0] = 1
predicted[predicted > 0] =1
#initialize counter variables
correct = 0
falsePositive = 0
falseNegative = 0
groundTruthTrueVoxels = 0
totalCheckedVoxels = 0
for x in actual:
	for y in x:
		for z in y:
			totalCheckedVoxels++
			if(actual[x,y,z]):
				groundTruthTrueVoxels++
				if(predicted[x,y,z]):
					correct++
				else:
					falseNegative++
			if(predicted[x,y,z] and not actual[x,y,z]):
				falsePositive++
falseAlarmRate = falsePositive/(totalCheckedVoxels - groundTruthTrueVoxels)
predictionRate = correct/groundTruthTrueVoxels
missRate = falseNegative/(totalCheckedVoxels - groundTruthTrueVoxels)
print 'far: ' + falseAlarmRate
print 'pred: ' + predictionRate
print 'miss: ' + missRate
