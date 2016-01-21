import ndio
import ndio.convert.png as ndpng
import ndio.remote.OCP as OCP
import numpy as np

def getRaw(x1,x2,y1,y2,z1,z2):
	"""
	Retrieve mitochondria image volume from neurodata given x,y,z parameters
	
	Arguments:
		x1/y1/z1: starting value for cut
		x2/y2/z2: ending cut value
	
	Returns: 
		3D Numpy Array of images (each image is in a 2D numpy array)
	"""
	oo = OCP()
	return oo.get_cutout('kasthuri11cc', 'image', x1, x2, y1, y2, z1, z2, resolution = 3)
	#mito_img = oo.get_cutout('kasthuri11cc', 'image', 694, 1794, 1750, 2640, 1004, 1154, resolution=3)


def getTruth(x1,x2,y1,y2,z1,z2):
	"""
	Retrieve mitochondria annotation image volume from neurodata given x,y,z parameters
	
	Arguments:
		x1/y1/z1: starting value for cut
		x2/y2/z2: ending cut value
	
	Returns: 
		3D Numpy Array of images (each image is in a 2D numpy array)
	
	"""
	oo = OCP()
	return oo.get_cutout('kasthuri2015_ramon_v1', 'mitochondria', x1, x2, y1, y2, z1, z2, resolution = 3)
	#mito_anno = oo.get_cutout(token, channel, 694, 1794, 1750, 2640, 1004, 1154, resolution=3)


def convertToPNG(npArray, target_directory):
	"""
	Converts 3D numpy array image volume into a bunch of pngs, dumps them in specified directory
	
	Arguments:
		npArray: 3D numpy array to save as png collection
		target_directory: String path of directory for images to be saved
	
	Returns: 
		None
	"""
	ndpng.export_png_collection(target_directory, npArray)

def convertToType(npArray,typeAsString):
	npArray = npArray.astype(typeAsString)
	return npArray

def visualThreshold(npArray, value):
	""" 
	Thresholds all pixel values greater than 0 in numPy array and sets them to given value 
	
	Arguments: 
		npArray: A numpy array
		value: int to set all values greater than zero to
	
	Returns:
		Numpy array 
	
	"""
	#npArray = npArray.astype('uint8')
	#print(mito_anno)
	npArray[npArray > 0] = value
	return npArray
	
def getIndices(intValue, npArray, condition):
	"""
	Gets the indices of an array where the values are equal/greater to intValue
	
	Arguments:
		intValue: Integer value to look for
		npArray: numpy array of an image/images
		condition: String of whether you want to look for values greater or equal to intValue
	
	Returns: 
		Numpy Array
	"""
	if condition=="equal":
		indices = np.where(npArray == intValue) #tuple where condition is true
	if condition=="greater":
		indices = np.where(npArray > intValue)
	return indices #array in which array[image # in stack][y value of image array][x value of image array]
	
def getxyMinMax(indexArray):
	"""Gets the min and maximum values for x,y in which indexArray[image # in stack][y value of image array][x value of image array]
	
	Arguments:
		indexArray: a 3D numpy array of indices in which 0th index is which image, 1st index corresponds to x-axis of image, 2nd index corresponds to y-axis of image
	
	Returns:
		Tuple of the minimums and maximums of x and y axis of image volume
	"""
	xMin = np.amin(indexArray[2])
	xMax = np.amax(indexArray[2])
	yMin = np.amin(indexArray[1])
	yMax = np.amax(indexArray[1])
	return (xMin,xMax,yMin,yMax)
	
def getCutValues():
	"""Retrives annotations, finds indices, and returns the maximum indices of mitochondria in the image stack"""
	mito_anno = getTruth(694, 1794, 1750, 2640, 1004, 1154)
	indices = getIndices(0, mito_anno, "greater")
	return getxyMinMax(indices)

def writeConfig(xyTuple):
	"""Creates a config file (config.py) and writes 4 values into it
	
	Arguments:
		xyTuple: a tuple of structure (xMin, xMax, yMin, yMax)
	"""
	
	outfile = open("config.py","w")
	outfile.write("X_MIN="+str(xyTuple[0])+"\n")
	outfile.write("X_MAX="+str(xyTuple[1])+"\n")
	outfile.write("Y_MIN="+str(xyTuple[2])+"\n")
	outfile.write("Y_MAX="+str(xyTuple[3]))
	outfile.close()
	return None
	

def main():
	"""Retrieves mitochondria annotations, visually thresholds them, converts them to pngs, and dumps them in a specified directory"""
	#writeConfig(getCutValues())
	print('Retrieving numpy array...')
	#mito_anno = getTruth(694, 1794, 1750, 2640, 1004, 1154)
	#mito_img = getRaw(694, 1794, 1750, 2640, 1004, 1154)
	mito_anno = ndio.convert.png.import_png_collection('bestdata/mito_anno_*')
	mito_anno = np.asarray(mito_anno)
	print('Converting numpy array to uint8')
	mito_anno = convertToType(mito_anno, "uint8")

	mito_anno = visualThreshold(mito_anno, 255)
	
	#indices = getIndices(255, mito_anno, "equal")
	#xMin, xMax, yMin, yMax = getxyMinMax(indices)
	#print(xMin,xMax,yMin,yMax)
	
	#oo=OCP()
	#rip = oo.get_cutout('kasthuri2015_ramon_v1', 'mitochondria', 694+xMin, 694+xMax, 1750+yMin, 1750+yMax, 1004, 1154, resolution=3)
	#rip = convertToType(rip, 'uint8')
	#rip = visualThreshold(rip, 255)

	print('Converting to png...')
	convertToPNG(mito_anno,'data/improved_annotations/improved_mito_anno_*')
if __name__ == '__main__':
    main()
