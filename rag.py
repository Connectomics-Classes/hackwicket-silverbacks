from skimage.future import graph
from skimage import segmentation, data, color
from scipy.misc import toimage
import numpy as np


def rag_mean_color(image, labels, connectivity=2, mode='distance',
                   sigma=255.0):
    rag = graph.RAG(labels, connectivity=connectivity)

    for n in rag:
        rag.node[n].update({'labels': [n],
                              'pixel count': 0,
                              'total color': np.array([0, 0, 0],
                                                      dtype=np.double)})

    for index in np.ndindex(labels.shape):
        current = labels[index]
        rag.node[current]['pixel count'] += 1
        rag.node[current]['total color'] += image[index]

    for n in rag:
        rag.node[n]['mean color'] = (rag.node[n]['total color'] /
                                       rag.node[n]['pixel count'])

    for x, y, d in rag.edges_iter(data=True):
        diff = rag.node[x]['mean color'] - rag.node[y]['mean color']
        diff = np.linalg.norm(diff)
        if mode == 'similarity':
            d['weight'] = math.e ** (-(diff ** 2) / sigma)
        elif mode == 'distance':
            d['weight'] = diff
        else:
            raise ValueError("The mode '%s' is not recognised" % mode)

    return rag

def rag_mean_intensity(image, labels, connectivity=2):

    rag = graph.RAG(labels, connectivity=connectivity)

    for n in rag:
        rag.node[n].update({'labels': [n],
                            'pixel count': 0,
                            'total intensity': 0})

    for index in np.ndindex(labels.shape):
        current = labels[index]
        rag.node[current]['pixel count'] += 1
        rag.node[current]['total intensity'] += image[index]

    for n in rag:
        if rag.node[n]['pixel count'] == 0:
            print 'node', n, ' has 0 labels'
            rag.node[n]['mean intensity'] = 0
        else:   
            rag.node[n]['mean intensity'] = (rag.node[n]['total intensity'] /
                                             rag.node[n]['pixel count'])

    for x, y, d in rag.edges_iter(data=True):
        diff = rag.node[x]['mean intensity'] - rag.node[y]['mean intensity']
        diff = abs(diff)
        d['weight'] = diff

    return rag 

def main():
    img = data.astronaut()
    labels = segmentation.slic(img, 400, 1)
    rag = rag_mean_color(img, labels)
    g_img = graph.draw_rag(labels, rag, img)
    toimage(g_img).show()
    raw_input('waiting')
    img = color.rgb2grey(data.astronaut())
    labels = segmentation.slic(img, 400, 1)
    rag = rag_mean_intensity(img, labels)
    g_img = graph.draw_rag(labels, rag, img)
    toimage(g_img).show()
    raw_input('waiting')

main()
