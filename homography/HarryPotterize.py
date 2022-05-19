import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac,computeH,computeH_norm,compositeH
import skimage.io
import skimage.color
from opts import get_opts
import matplotlib.pyplot as plt
from planarH import computeH_norm
from scipy.ndimage.interpolation import rotate
#Import necessary functions
#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
# rot_hp_cover = rotate(hp_cover, angle = 15)

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
# matches, locs1, locs2 = matchPics(cv_cover, cv_cover, opts)
# matches, locs1, locs2 = matchPics(hp_cover, rot_hp_cover, opts)

x1 = locs1[matches[:, 0]]
x2 = locs2[matches[:, 1]]
x1[:, [1,0]] = x1[:, [0,1]]
x2[:, [1,0]] = x2[:, [0,1]]

bestH2to1, inliers = computeH_ransac(x1, x2, opts)
# print('H: \n{}'.format(bestH2to1))


# cv_cover_shifted = cv2.warpPerspective(cv_cover, np.linalg.inv(bestH2to1), (cv_desk.shape[1], cv_desk.shape[0]))
# plt.imshow(cv_cover_shifted)
# plt.show()

# print("stop")

hp_cover = cv2.resize(hp_cover, dsize=(cv_cover.shape[1], cv_cover.shape[0]))
composite_img = compositeH(bestH2to1, hp_cover, cv_desk)

# hp_shifted = cv2.warpPerspective(hp_cover,bestH2to1, (cv_desk.shape[1], cv_desk.shape[0]), flags=cv2.WARP_INVERSE_MAP)
# plt.imshow(hp_shifted)
# plt.show()

# print("stop")


plt.imshow(composite_img)
plt.show()




# cv_shifted = cv2.warpPerspective(cv_cover, bestH2to1, (cv_desk.shape[1], cv_desk.shape[0]), flags=cv2.WARP_INVERSE_MAP)
# plt.imshow(cv_shifted)
# plt.show()

# x1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# x2 = x1
# x2 = x1*2
# x2 = np.array([[-1, -1],[-1, 0],[0, -1],[0, 0]])
# H2to1test = computeH(x1, x2)
# H2to1test /= H2to1test[-1,-1]
# print('H: \n{}'.format(H2to1test))
# print("stop")

# x1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# x2 = x1
# x2 = x1*2
# x2 = np.array([[-1, -1],[-1, 0],[0, -1],[0, 0]])
# H2to1test = computeH_norm(x1, x2)
# print('H: \n{}'.format(H2to1test))
# print("stop")


# x1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# x2 = x1
# x2 = x1*2
# x2 = np.array([[-1, -1],[-1, 0],[0, -1],[0, 0]])
# plotMatches(cv_cover, cv_desk, matches, locs1, locs2)