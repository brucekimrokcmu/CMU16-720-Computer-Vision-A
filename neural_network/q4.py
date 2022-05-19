# import skimage
# import skimage.measure
# import skimage.color
# import skimage.restoration
# import skimage.filters
# import skimage.morphology
# import skimage.segmentation

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage
import skimage.io
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    # for img in os.listdir('../images'):
    #     im1 = rgb2gray(skimage.io.imread(os.path.join('../images',img)))
    #     im1 = skimage.img_as_float(im1)   
        
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image < thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around sbboxesegmented coins
            minr, minc, maxr, maxc = region.bbox
            bboxes.append(region.bbox)
    #         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                 fill=False, edgecolor='red', linewidth=2)
    #         ax.add_patch(rect)

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    ##########################

    return bboxes, bw







