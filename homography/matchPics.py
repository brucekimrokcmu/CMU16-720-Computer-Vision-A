import numpy as np
import cv2
import skimage.color
from skimage.feature import corner
from skimage.feature.util import plot_matches
from helper import briefMatch, plotMatches
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
    #I1, I2 : Images to match
    #opts: input opts
    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

    #Convert Images to GrayScale
    img1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)

    #Detect Features in Both Images
    intpts1 = corner_detection(img1, sigma)
    intpts2 = corner_detection(img2, sigma)

    #Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(img1, intpts1)
    desc2, locs2 = computeBrief(img2, intpts2)

    #Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2
