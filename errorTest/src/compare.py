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
#set all mitochondria to detected
actual[actual > 0] = 100
predicted[predicted > 0] =100
