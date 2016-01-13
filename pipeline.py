import ndio.convert.png
import numpy as np
import sklearn.ensemble as skl
from skimage import segmentation, transform, color
from skimage.future import graph
from scipy.misc import toimage
import rag2 as graph_custom # edited skimage.future.graph module

def smartDownSample(img, nsegs, compactness):
    labels = segmentation.slic(img, nsegs, compactness, multichannel = False, enforce_connectivity = True)
    rag = graph_custom.rag_mean_int(img, labels)
    new_labels = graph.cut_normalized(labels, rag, .5)
    rag = graph_custom.rag_mean_int(img, new_labels)
    nsegs = len(np.unique(labels)) # Sometimes it differs
    counts = np.zeros([nsegs])
    totals = np.zeros([nsegs])
    for (x,y), val in np.ndenumerate(labels):
        counts[val] += 1
        totals[val] += img[x][y]
    averages = np.floor(np.divide(totals, counts))
    new_img = np.empty([len(img), len(img[0])])
    for (x,y), val in np.ndenumerate(labels):
        new_img[x][y] = averages[val]
    g_img = graph_custom.draw_rag(new_labels, rag, color.gray2rgb(new_img))
    toimage(g_img).show()
    raw_input('waiting...')
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

    train = mito_img[70]
    train_truth = mito_anno[70]

    test = mito_img[71]
    test_truth = mito_anno[71]

    X_train = train.flatten().reshape(n_rows * n_cols, 1)
    y_train = train_truth.flatten()

    X_test = test.flatten().reshape(n_rows * n_cols, 1)[5000:5100]
    y_test = test_truth.flatten()[5000:5100]

    smartDownSample(train, 10000, .5) 

#    THIS IS VERY SLOW, even with one image training and one image predicting.
#    Trying the graph cuts strategy to create objects to classify 
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
