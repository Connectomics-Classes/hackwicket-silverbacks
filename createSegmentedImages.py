import sys, random
import numpy as np
import ndio.convert.png
import ndio.remote.OCP as OCP

import cutImageNAnnotations as cutImg


def getConfig():
	"""Returns x-min,x-max,y-min,y-max in tuple form in respective order"""
	try:
		import config
		return (config.X_MIN, config.X_MAX, config.Y_MIN, config.Y_MAX)
	except ImportError:
		maxes = cutImg.getCutValues()
		cutImg.writeConfig(maxes) #creates config.py
		return maxes
		

def getRaw(db = False):
    # Careful of hardcoded filepath    
    if db == False:
        vol = np.array([])
        data = ndio.convert.png.import_png_collection('newdata/mito_data_*')
        for img in data:
            vol = np.dstack([vol, img]) if vol.size else img
        return vol
    else:
		boundaries = getConfig()
		oo = OCP()
		#added +12 because first 10 images in stack aren't even annotated
		return oo.get_cutout('kasthuri11cc', 'image', 694 + boundaries[0], 694 + boundaries[1], 1750 + boundaries[2], 1750 + boundaries[3], 1004, 1154, resolution = 3) 


def getTruth(db = False):
    # Careful of hardcoded filepath    
    if db == False:
        vol = np.array([])
        data = ndio.convert.png.import_png_collection('newdata/mito_anno_*')
        for img in data:
            vol = np.dstack([vol, img]) if vol.size else img
        return vol
    else:
		boundaries = getConfig()
		oo = OCP()
		#added +12 because first 10 images in stack aren't even annotated
		return oo.get_cutout('kasthuri2015_ramon_v1', 'mitochondria', 694 + boundaries[0], 694 + boundaries[1], 1750 + boundaries[2], 1750 + boundaries[3], 1004, 1154, resolution = 3)

def main():
	print("Retrieving data")
	mito_img = getRaw(db=True)
	mito_anno = getTruth(db=True)
	
	mito_anno = cutImg.convertToType(mito_anno, 'uint8') #uncomment/comment if you want to visually threshold the annotated mitochondr images
	mito_anno = cutImg.visualThreshold(mito_anno, 255) #uncomment/comment if you want to visually threshold the annotated mitochondr images

	print('Converting to png...')
	cutImg.convertToPNG(mito_anno,'bestdata/mito_anno_*')
	cutImg.convertToPNG(mito_img, 'bestdata/mito_data_*')
if __name__ == '__main__':
    main()
