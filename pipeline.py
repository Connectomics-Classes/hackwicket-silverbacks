from skimage import draw, measure, segmentation, util, color
import numpy as np
import ndio.convert.png
import sklearn.ensemble as skl
from skimage.future import graph
from scipy.misc import toimage
import rag2 as graph_custom # edited skimage.future.graph module

def smartDownSample(img, nsegs, compactness, threshold):
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
    return new_img

def getRaw():
    # Careful of hardcoded filepath    
    return ndio.convert.png.import_png_collection('data/mito_img_*')

def getTruth():
    # Careful of hardcoded filepath    
    return ndio.convert.png.import_png_collection('data/mito_anno_*')

def main():
    print "Getting raw data from disk..."
    mito_img = getRaw()

    print "Getting annotations from disk..."
    mito_anno = getTruth()

    print "Processing features..."
    n_rows = len(mito_img[0])
    n_cols = len(mito_img[0][0])

    print "Initializing oversegmentation / threshold cutting algorithm..."
    n_segments = 50000 # (approx) the number of superpixels from SLIC
    compactness = .2 # how much SLIC favors shape vs color (high = more square)
    threshold = .1 # gradient threshold for threshold cut
    smartDownSample(mito_img[70], 50000, .19, .085) 

#    Outline of ML stuff. Tried pixel classification...
#    THIS IS VERY SLOW, even with one image training and one image predicting.
#    Trying the graph cuts strategy to create objects to classify 
#
#    train = mito_img[70]
#    train_truth = mito_anno[70]
#
#    test = mito_img[71]
#    test_truth = mito_anno[71]
#
#    X_train = train.flatten().reshape(n_rows * n_cols, 1)
#    y_train = train_truth.flatten()
#
#    X_test = test.flatten().reshape(n_rows * n_cols, 1)[5000:5100]
#    y_test = test_truth.flatten()[5000:5100]
#
#    print "Initializing random forest..."
#    forest = skl.RandomForestClassifier(max_depth = 10, verbose = 1)
#
#    print "Learning random forest..."
#    forest.fit(X_train, y_train)
#    y_pred = forest.predict(X_test)
#    for i in y_pred:
#        print i

if __name__ == '__main__':
    main()
