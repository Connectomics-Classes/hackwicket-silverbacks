from skimage import draw, measure, segmentation, util, color
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

def smartDownSample(img, nsegs, compactness, threshold):
    # Segment using SLIC
    labels = segmentation.slic(img, nsegs, compactness, multichannel = False, enforce_connectivity = True)
    # Make a Region Adjacency Graph of the segmentation
    rag = graph_custom.rag_mean_int(img, labels)
    # Cut this RAG based on threshold
    new_labels = graph.cut_threshold(labels, rag, threshold)
    # Make a new graph based on the new segmentation (post cut)
    # not using right now... rag = graph_custom.rag_mean_int(img, new_labels)
    
    new_labels = np.add(new_labels, np.ones(new_labels.shape, dtype=np.int8))
    return new_labels 

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

def extract_features(img, labels):
    n_features = 3
    n_samples = labels.max()
    props = measure.regionprops(labels, intensity_image = img)
    X = np.empty([n_samples, n_features])
    for i in range(labels.max()):
        tmp = props[i]
        X[i][0] = tmp['mean_intensity'] 
        X[i][1] = tmp['area']
        if props[i]['minor_axis_length'] != 0:
            X[i][2] = tmp['major_axis_length'] / tmp['minor_axis_length']
        else:
            X[i][2] = 0
    return X

def extract_classes(truth, labels):
    n_samples = labels.max()
    y = np.zeros([n_samples,])
    truth[truth > 0]  = 255
    props = measure.regionprops(labels, intensity_image = truth)
    for i in range(labels.max()):
        if props[i]['mean_intensity'] > 127:
            y[i] = 1
    return y

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

    labels = smartDownSample(mito_img[70], n_segments, compactness, threshold) 
    p_labels = smartDownSample(mito_img[71], n_segments, compactness, threshold)
    demo(mito_img[70], n_segments, compactness, threshold)
    print "Processing features..."
    X_train = extract_features(mito_img[70], labels)
    y_train = extract_classes(mito_anno[70], labels)
    X_test = extract_features(mito_img[71], p_labels)
    print "Initializing random forest..."
    forest = skl.RandomForestClassifier(max_depth = 10)
    print "Learning random forest..."
    forest.fit(X_train, y_train)
    print "Making predictions..."
    y_pred = forest.predict(X_test)
    y_pred_img = np.empty(mito_anno[70].shape)
    for index in np.ndindex(y_pred_img.shape):
        y_pred_img[index] = y_pred[p_labels[index] - 1]
    y_pred_img[ y_pred_img > 0 ] = 255 
    print "Showing true img..."
    toimage(mito_img[71]).show()
    print "Showing predicted img..."
    toimage(y_pred_img).show()
        

if __name__ == '__main__':
    main()
