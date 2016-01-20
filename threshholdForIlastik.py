import ndio
import ndio.convert.png as ndpng
import ndio.remote.OCP as OCP
import numpy as np

def getRaw(ndio = False):
    # Careful of hardcoded filepath    
    if ndio == False:
        mito_img = ndpng.import_png_collection('data/mito_img_*')
        mito_img = np.array(mito_img)
        return mito_img
    else:
        oo = OCP()
        return oo.get_cutout('kasthuri11cc', 'image', 694, 1794, 1750, 2640, 1004, 1154, resolution=3)

def getTruth(ndio = False):
    # Careful of hardcoded filepath    
    if ndio == False:
		mito_anno = ndpng.import_png_collection('data/mito_anno_*')
		mito_anno = np.array(mito_anno)
		return mito_anno
    else:
        oo = OCP()
        return oo.get_cutout('kasthuri2015_ramon_v1', 'mitochondria', 694, 1794, 1750, 2640, 1004, 1154, resolution=3)

def convertToPNG(npArray, target_directory):
	ndio.convert.png.export_png_collection(target_directory, npArray)

def convertToType(npArray,typeAsString):
	npArray = npArray.astype(typeAsString)
	return npArray

def visualThreshold(npArray, value):
	""" Thresholds all pixel values greater than 0 in numPy array and sets them to value """
	#npArray = npArray.astype('uint8')
	#print(mito_anno)
	npArray[npArray > 0] = value
	return npArray
	
def getIndices(intValue, npArray):
	indices = np.where(npArray == intValue) #tuple where condition is true
	return indices #array in which array[image # in stack][y value of image array][x value of image array]
	
def getxyMinMax(indexArray):
	"""Gets the min and maximum values for x,y in which indexArray[image # in stack][y value of image array][x value of image array]"""
	xMin = np.amin(indexArray[2])
	xMax = np.amax(indexArray[2])
	yMin = np.amin(indexArray[1])
	yMax = np.amax(indexArray[1])
	return xMin,xMax,yMin,yMax
	
def main():
	print('Retrieving numpy array...')
	mito_anno = getTruth(ndio = False)
	print('Converting numpy array to uint8')
	mito_anno = convertToType(mito_anno, "uint8")
	#mito_anno = mito_anno.astype('uint8')
	#print(mito_anno)
	mito_anno = visualThreshold(mito_anno, 255)
	
	indices = getIndices(255, mito_anno)
	#print(indices)
	xMin, xMax, yMin, yMax = getxyMinMax(indices)
	print(xMin,xMax,yMin,yMax)
	
	oo=OCP()
	rip = oo.get_cutout('kasthuri2015_ramon_v1', 'mitochondria', 694+xMin, 694+xMax, 1750+yMin, 1750+yMax, 1004, 1154, resolution=3)
	rip = convertToType(rip, 'uint8')
	rip = visualThreshold(rip, 255)

	print('Converting to png...')
	convertToPNG(rip,'data/improved_annotations/improved_mito_anno_*')
if __name__ == '__main__':
    main()
