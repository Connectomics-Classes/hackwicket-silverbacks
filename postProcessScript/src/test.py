#!/usr/bin/python

#imports
import numpy as np
import cv
import cv2
import sys
from mayavi import mlab
from mayavi.mlab import *

#load the numpy into the program
originalData = np.load('../data/test.npy')

###########this is for testing purposes
######this is for erosion & dilation with the trackbar
def nothing(x):
	pass

#get the data into a usable form
testArr = originalData[0, :, :, 1]
testArr = testArr.copy()
#apply erosion & dilation filtering
cv2.namedWindow('filterMenu')
cv2.createTrackbar('erosion','filterMenu',1,255, nothing)
cv2.createTrackbar('dilation','filterMenu',1,255, nothing)
#apply threshold filtering
cv2.createTrackbar('lowRange','filterMenu',1, 100, nothing)
cv2.createTrackbar('highRange','filterMenu',100, 100, nothing)
while(1):
	ero = cv2.getTrackbarPos('erosion','filterMenu')
	dil = cv2.getTrackbarPos('dilation', 'filterMenu')
	kernel = np.ones((ero, ero), np.uint8)
	erodedArr = cv2.erode(testArr, kernel)
	kernel = np.ones((dil,dil), np.uint8)
	dilatedArr = cv2.dilate(erodedArr, kernel)
	#turn decimals to percentages
	restrictedArr = cv2.inRange(dilatedArr, cv2.getTrackbarPos('lowRange', 'filterMenu')/100., cv2.getTrackbarPos('highRange', 'filterMenu')/100.)
	cv2.imshow('testImg',restrictedArr)
	if(cv2.waitKey(25) == ord('a')):
		break
#this is where images would get stitched together

#display images
newArr = restrictedArr.copy()
newArr[newArr > 0] = 100
#add the z dim
dispArr = newArr[:,:,np.newaxis]
print dispArr.shape
#test display

mlab.contour3d(dispArr)
mlab.show()
		


