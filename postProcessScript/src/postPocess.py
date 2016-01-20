#!/usr/bin/python

#imports
import numpy as np
import cv
import cv2
import sys
import glob
from mayavi import mlab
from mayavi.mlab import *

def morphOps(probMap):
	workableMap = probMap.copy()
	#apply erosion & dilation filtering, rids photo of stray vescicle detections
	workableMap = eroDilFilter(workableMap, 5, 3) #4 and 1 experimentally determined
	#restrict array to include only above 70% confidence
	restrictedArr = cv2.inRange(workableMap,.7, 1)
	#change all remaining values to 100 	
	restrictedArr[restrictedArr > 0] = 100
	return restrictedArr

def loadNpyVolume():
	print 'loading files from data directory...'
	numpyVolume = []
	#for all of the numpy arrays in the current directory
	for numpyArr in glob.glob('../data/*.np[yz]'):
		probMap = np.load(numpyArr)
		#if the numpy is 3d
		if(len(probMap.shape) == 3):
			#split and add sub numpys to volume
			#reorder parameters for looping
			np.rollaxis(probMap, 2)
			for subMap in probMap:
				#add all subArrs in the 3d npy to the volume
				print subMap.shape
				numpyVolume.append(subMap)
		#if the numpy is 2d
		elif(len(probMap.shape) == 2):
			numpyVolume.append(probMap)
		#if the numpy doesnt make sense
		else:
			print 'Error: Npy data format not recognized'
	return numpyVolume

def eroDilFilter(array,ero, dil):
	kernel = np.ones((ero, ero), np.uint8)
	erodedArr = cv2.erode(array,kernel)
	kernel = np.ones((dil,dil), np.uint8)
	dilatedArr = cv2.dilate(array, kernel)
	return dilatedArr

#load the numpy into the program
numpyVolume = loadNpyVolume()
#instantiate list of array for display
stackableList = []
#perform morph ops to clean up data
print 'cleaning up data...'
for probMap in numpyVolume:
	displayArr = morphOps(probMap)
	#add array to display list
	stackableList.append(displayArr)

#stitch arrays together for display
print 'generating 3d volume...'
finalVolume = np.dstack(stackableList)

#display arrays
mlab.contour3d(finalVolume)
mlab.show()
