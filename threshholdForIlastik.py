import ndio
import ndio.convert.png as ndpng
import ndio.remote.OCP as OCP
import numpy as np

def getRaw(ndio = False):
    # Careful of hardcoded filepath    
    if ndio == False:
        return ndio.convert.png.import_png_collection('data/mito_img_*')
    else:
        oo = OCP()
        return oo.get_cutout('kasthuri11cc', 'image', 694 + 538, 694 + 844, 1750 + 360, 1750 + 520, 1004, 1154, resolution = 3)

def getTruth(ndio = False):
    # Careful of hardcoded filepath    
    if ndio == False:
        return ndpng.import_png_collection('data/mito_anno_*')
    else:
        oo = OCP()
        return oo.get_cutout('kasthuri2015_ramon_v1', 'mitochondria', 694, 1794, 1750, 2640, 1004, 1154, resolution=3)

def convertToPNG(npArray):
	ndio.convert.png.export_png_collection('data/improved_annotations/improved_mito_anno_*', npArray)

def main():
	print('Retrieving numpy array...')
	mito_anno= getTruth(ndio = True)
	print('Converting numpy array to uint8')
	mito_anno = mito_anno.astype('uint8')
	#print(mito_anno)
	mito_anno[mito_anno > 0] = 255
	
	#for a in range(len(mito_anno)):
		##print((mito_anno[a]))
		#for b in range(len(mito_anno[a])):
			#print(mito_anno[a][b])
			##for c in range(len(mito_anno[a][b])):
				###print a,b,c
				##if mito_anno[a][b][c] == 0:
					##mito_anno[a][b][c] = 100;
	print('Converting to png...')
	convertToPNG(mito_anno)
if __name__ == '__main__':
    main()
