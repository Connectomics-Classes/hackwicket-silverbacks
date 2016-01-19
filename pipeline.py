from skimage import draw, measure, segmentation, util, color, exposure
import sys
import numpy as np
import ndio.convert.png
import ndio.remote.OCP as OCP
import sklearn.ensemble as skl
from skimage.future import graph
from scipy.misc import toimage
import rag2 as graph_custom # edited skimage.future.graph module

def demo(img, nsegs, compactness, threshold):
    # Segment using SLIC
    labels = segmentation.slic(img, nsegs, compactness, multichannel = False, enforce_connectivity = True)
    # Make a Region Adjacency Graph of the segmentation
    rag = graph_custom.rag_mean_int(img, labels)
    # Cut this RAG based on threshold
    new_labels = graph.cut_threshold(labels, rag, threshold)
    # Make a new graph based on the new segmentation (post cut)
    # not using right now... rag = graph_custom.rag_mean_int(img, new_labels)
    # Average the pixel color in every segmented region, into new_img
    nsegs = len(np.unique(new_labels))
    counts = np.zeros([nsegs])
    totals = np.zeros([nsegs])
    for (x,y), val in np.ndenumerate(new_labels):
        counts[val] += 1
        totals[val] += img[x][y]
    averages = np.floor(np.divide(totals, counts))
    new_img = np.empty([len(img), len(img[0])])
    for (x,y), val in np.ndenumerate(new_labels):
        new_img[x][y] = averages[val]
    # Show the images
    toimage(img).show()
    print "Original Image"
    slic_img = segmentation.mark_boundaries(new_img, labels)
    toimage(slic_img).show()
    print "Image after only SLIC algorithm"
    seg_img = segmentation.mark_boundaries(new_img, new_labels)
    print "Image after threshold cut"
    toimage(seg_img).show()


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
        return ndio.convert.png.import_png_collection('data/mito_anno_*')
    else:
        oo = OCP()
        return oo.get_cutout('kasthuri2015_ramon_v1', 'mitochondria', 694 + 538, 694 + 844, 1750 + 360, 1750 + 520, 1004, 1154, resolution = 3)


def extract_features(im_vol, label_vol):
    n_features = 3
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
            tX[j-1][1] = tmp['area']
            if tmp['minor_axis_length'] != 0:
                tX[j-1][2] = tmp['major_axis_length'] / tmp['minor_axis_length']
            else:
                tX[j-1][2] = 0
        X = np.vstack([X, tX]) if X.size else tX
    return X


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


def equalize_volume(volume):
    for ra in volume:
        ra = (np.floor(exposure.equalize_hist(ra) * 256)).astype('uint8')
    return volume
        

def main():
    print "Getting raw data..."
    mito_img = getRaw(ndio = True)

    print "Getting annotations..."
    mito_anno = getTruth(ndio = True)

    n_rows = len(mito_img[0])
    n_cols = len(mito_img[0][0])

    print "Initializing oversegmentation / threshold cutting algorithm..."
    n_segments = int(sys.argv[1]) # (approx) the number of superpixels from SLIC
    compactness = float(sys.argv[2]) # how much SLIC favors shape vs color (high = more square)
    threshold = float(sys.argv[3]) # gradient threshold for threshold cut

    # Implementing histogram equalization for better contrast!
    train = equalize_volume(mito_img[100:149])
    test = equalize_volume(mito_img[97:100])

    train_truth = mito_anno[100:149]
    test_truth = mito_anno[97:100]

    train_labels = smartDownSample(train, n_segments, compactness, threshold) 
    test_labels = smartDownSample(test, n_segments, compactness, threshold)

   # demo(mito_img[70], n_segments, compactness, threshold)

    print "Processing features..."
    X_train = extract_features(train, train_labels)
    y_train = extract_classes(train_truth, train_labels)
    X_test = extract_features(test, test_labels)
    print len(X_train), len(y_train), len(X_test)

    print "Initializing random forest..."
    forest = skl.RandomForestClassifier(max_depth = 10)

    print "Learning random forest..."
    forest.fit(X_train, y_train)

    print "Making predictions..."
    y_guess = forest.predict(X_test)

    print "Showing images..."
    j = 0
    for i in range(len(test_labels)):
        img = np.empty(test_labels[i].shape)
        for k in range(1, test_labels[i].max()):
            img[test_labels[i] == k] = y_guess[j]
            j = j + 1
        toimage(test[i]).show()
        label_img = np.empty(test_labels[i].shape)
        for g in range(test_labels[i].max()):
            props = measure.regionprops(test_labels[i], intensity_image = test[i])
            label_img[test_labels[i] == g] = props[g - 1]['mean_intensity']
        toimage(label_img).show()
        toimage(img).show()
        toimage(test_truth[i]).show()
        print j
#    y_guess_imgs = np.empty(test.shape)
#    for index in np.ndindex(test.shape):
#        y_guess_imgs[index] = y_guess[test_labels[index] - 1]
#    y_guess_imgs[ y_guess_img > 0 ] = 255 
#    print "Showing true img..."
#    for img in y_guess_imgs:
#        toimage(img).show()
        

if __name__ == '__main__':
    main()
