import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from numpy.lib import median

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform


from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


for img in os.listdir('../images'):
    im1 = rgb2gray(skimage.img_as_float(skimage.io.imread(os.path.join('../images',img))))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        # box = bw[minr:maxr, minc:maxc]
        # plt.imshow(box)
        # plt.show()
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    height = []
    for bbox in bboxes:    
        minr, minc, maxr, maxc = bbox
        height.append(maxr - minr)
    median_height = np.median(height)
    ##########################

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    row_list = []
    current_row = []
    last_minr = bboxes[0][0]
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        if np.absolute(minr-last_minr) < median_height:
            current_row.append(bbox)
            last_minr = minr
        else:
            row_list.append(current_row)
            current_row = []
            last_minr = minr
    row_list.append(current_row)
    current_row = []

    for row in row_list:
        row.sort(key=lambda x:x[1])
    # for row in row_list:
    #     for bbox in row:
    #         minr, minc, maxr, maxc = bbox
    #         plt.imshow(bw[minr:maxr,minc:maxc])
    #         plt.show()
    flat_row = []
    flat_row_list = []
    for row in row_list:
        for bbox in row:
            minr, minc, maxr, maxc = bbox
            img = bw[minr:maxr,minc:maxc]
            img = skimage.transform.resize(img, (20,20))
            img = 1-np.pad(img, (6,6))
            img = skimage.morphology.erosion(img)
            # plt.imshow(img)
            # plt.show()
            flat_row.append((np.transpose(img)).flatten())
            # plt.imshow(img)
            # plt.show
        flat_row_list.append(flat_row)
        flat_row = []

    ##########################
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    for flat_list in flat_row_list:
        poo = ''
        for xb in flat_list:
            hl1 = forward(np.expand_dims(xb,axis=0), params, name='layer1', activation=sigmoid)
            probs = forward(hl1, params, name='output', activation=softmax) 
            
            pred = np.argmax(probs, axis=1)
            pred2let = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            poo += pred2let[pred[0]]
        print(poo)
    print()
    ##########################
