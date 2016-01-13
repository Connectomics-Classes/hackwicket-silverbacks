#!/usr/bin/python

#imports
import numpy as np
import cv
import cv2
import sys

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
while(1):
	ero = cv2.getTrackbarPos('erosion','filterMenu')
	dil = cv2.getTrackbarPos('dilation', 'filterMenu')
	kernel = np.ones((ero, ero), np.uint8)
	erodedArr = cv2.erode(testArr, kernel)
	kernel = np.ones((dil,dil), np.uint8)
	dilatedArr = cv2.dilate(erodedArr, kernel)
	cv2.imshow('testImg', dilatedArr)
	cv2.waitKey(0)


