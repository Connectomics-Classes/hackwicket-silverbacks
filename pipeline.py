from skimage import draw, measure, segmentation, util, color, exposure
import sys, random
import numpy as np
import ndio.convert.png
import ndio.remote.OCP as OCP
import sklearn.ensemble as skl
from skimage.future import graph
from scipy.misc import toimage
import rag2 as graph_custom # edited skimage.future.graph module
import cutImageNAnnotations as cutImg


# Input: vol - the 3D image volume
#        nsegs - approx. number of superpixels to create
#        compactness - weight of spatial distance in SLIC (as opposed to intensity distance)
#        threshold - graph cutting threshold (0 to 1)
# Output: volume of labels assigned to different objects in images
def smartDownSample(vol, nsegs, compactness, threshold):
    label_vol = np.empty(vol.shape)
    for i in range(len(vol)):
        # Segment using SLIC
        labels = segmentation.slic(vol[i], nsegs, compactness, multichannel = False, enforce_connectivity = True)
        # Make a Region Adjacency Graph of the segmentation
        rag = graph_custom.rag_mean_int(vol[i], labels)
        # Cut this RAG based on threshold
        new_labels = graph.cut_threshold(labels, rag, threshold)
        # Make a new graph based on the new segmentation (post cut)
        # not using right now... rag = graph_custom.rag_mean_int(img, new_labels)
        
        new_labels = np.add(new_labels, np.ones(new_labels.shape, dtype=np.int8))
        label_vol[i] = new_labels
    return label_vol.astype('uint8')


def getConfig():
	"""Returns x-min,x-max,y-min,y-max in tuple form in respective order"""
	try:
		import config
		return (config.X_MIN, config.X_MAX, config.Y_MIN, config.Y_MAX)
	except ImportError:
		maxes = cutImg.getCutValues()
		cutImg.writeConfig(maxes) #creates config.py
		return maxes
		
# Input: db - True if pulling data directly from ndio, False if local
# Output: the raw image volume
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


# Input: db - True if pulling data directly from ndio, False if local
# Output: the annotation volume
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


# Input: im_vol - the image volume of EM data
#        label_vol - the labels given to objects in EM data (from smartDownSample) 
# Output: the feature array X
def extract_features(im_vol, label_vol):
    n_features = 4
    X = np.array([])
    if len(im_vol) != len(label_vol):
        print 'volumes not the same z dimension'
    for i in range(len(im_vol)):
        n_samples = label_vol[i].max() - 1
        props = measure.regionprops(label_vol[i], intensity_image = im_vol[i])
        tX = np.empty([n_samples, n_features])
        for j in range(1, label_vol[i].max()):
            tmp = props[j-1]
            tX[j-1][0] = tmp['mean_intensity'] 
            tX[j-1][1] = tmp['eccentricity']
            tX[j-1][2] = tmp['convex_area']
            tX[j-1][3] = tmp['area']
        X = np.vstack([X, tX]) if X.size else tX
    return X

# Input: im_vol - the image volume of EM data
#        label_vol - the labels given to objects in EM data (from smartDownSample)
# Output: a vector of the classes (mito or not mito) given to each object (according to label_vol)
def extract_classes(im_vol, label_vol):
    y = np.zeros([0,])
    if len(im_vol) != len(label_vol):
        print 'volumes not the same z dimension'
    for i in range(len(im_vol)):
        n_samples = label_vol[i].max()
        im_vol[i][im_vol[i] > 0] = 255
        props = measure.regionprops(label_vol[i], intensity_image = im_vol[i])
        for j in range(1, label_vol[i].max()):
            if props[j-1]['mean_intensity'] > 150:
                y = np.append(y, 1)
            else:
                y = np.append(y, 0)
    return y

# Input: volume - an image volume
# Output: the image volume after histogram equalization
def equalize_volume(volume):
    for ra in volume:
        ra = (np.floor(exposure.equalize_hist(ra) * 256)).astype('uint8')
    return volume
        

def main():
    print "Getting raw data..."
    mito_img = getRaw(db = False)

    print "Getting annotations..."
    mito_anno = getTruth(db = False)
    

    n_rows = len(mito_img[0])
    n_cols = len(mito_img[0][0])

    print "Initializing oversegmentation / threshold cutting algorithm..."
    n_segments = int(sys.argv[1])
    compactness = float(sys.argv[2])
    threshold = float(sys.argv[3])

    train = equalize_volume(mito_img[70:150])
    test = equalize_volume(mito_img[30:70])

    train_truth = mito_anno[70:150]
    test_truth = mito_anno[30:70]

    nri = int(sys.argv[4])
    y_pred = [0] * nri

    for k in range(nri):
        r1, r2, r3 = random.randrange(-100, 100), random.uniform(-.1, .1,), random.uniform(-.05, .05)
        train_labels = smartDownSample(train, n_segments + r1, compactness + r2, threshold + r3) 
        test_labels = smartDownSample(test, n_segments + r1, compactness + r2, threshold + r3)

        print "Processing features..."
        X_train = extract_features(train, train_labels)
        y_train = extract_classes(train_truth, train_labels)
        X_test = extract_features(test, test_labels)


        print "Initializing random forest..."
        forest = skl.RandomForestClassifier(n_estimators = 200, max_depth = 100)

        print "Learning random forest..."
        forest.fit(X_train, y_train)

        print "Making predictions..."
        y_guess = forest.predict(X_test)

        # Move from prediction vector to prediction pictures
        j = 0
        y_pred[k] = np.zeros(test_labels.shape)
        for i in range(len(test_labels)):
            for g in range(1, test_labels[i].max()):
                y_pred[k][i][test_labels[i] == g] = y_guess[j]
                j = j + 1

#    print "Showing images..."
    y_pred_or = np.logical_or.reduce(y_pred)
#    for i in range(len(test_labels)):
#        toimage(test[i]).show()
#        label_img = np.empty(test_labels[i].shape)
#        for g in range(test_labels[i].max()):
#            props = measure.regionprops(test_labels[i], intensity_image = test[i])
#            label_img[test_labels[i] == g] = props[g - 1]['mean_intensity']
#        toimage(y_pred_or[i]).show()
#        toimage(label_img).show()
#        toimage(test_truth[i]).show()
    np.save('actual.npy', test)
    print X_test.shape
    np.save('predicted.npy', y_pred_or)
    np.save('truth.npy', test_truth)
    print y_pred_or.shape
        

if __name__ == '__main__':
    main()
