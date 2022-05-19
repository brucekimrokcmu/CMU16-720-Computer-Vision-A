import matplotlib
import matplotlib.pyplot
import numpy as np
import cv2
import scipy
from matchPics import matchPics
from opts import get_opts
from helper import briefMatch, plotMatches
from helper import computeBrief
from helper import corner_detection

opts = get_opts()
ratio = opts.ratio
sigma = opts.sigma

#Q2.1.6
#Read the image and convert to grayscale, if necessary

cv_cover = cv2.imread('../data/cv_cover.jpg')
img = cv2.cvtColor(cv_cover, cv2.COLOR_RGB2GRAY)

hist = []
for i in range(36):
	#Rotate Image
    rot_img = scipy.ndimage.rotate(img, 10*i)

	#Compute features, descriptors and Match features
    inputs1 = corner_detection(rot_img, sigma)
    desc1, locs1 = computeBrief(rot_img, inputs1)

    inputs2 = corner_detection(img, sigma)
    desc2, locs2 = computeBrief(img, inputs2)
    
    matches = briefMatch(desc1, desc2, ratio)
    count = matches.shape[0]
    # print(count)
	#Update histogram

    hist.append(count)
    # print(hist)
    # plotMatches(rot_img, img, matches, locs1, locs2, i)
	# pass # comment out when code is ready

#Display histogram

rot = np.arange(0, 360, 10)

matplotlib.pyplot.bar(rot, hist)
matplotlib.pyplot.show()