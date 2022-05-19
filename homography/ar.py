import numpy as np
import cv2
from loadVid import loadVid 
from planarH import compositeH, computeH_norm, computeH_ransac, composite_images
from matchPics import matchPics
from opts import get_opts
import matplotlib.pyplot as plt

#Import necessary functions
opts = get_opts()

# video[frame][H][W][Channel]

book = loadVid('../data/book.mov')
ar_source = loadVid('../data/ar_source.mov')
# plt.imshow(book[500])
# plt.show()

# print("stop")
cv_cover = cv2.imread('../data/cv_cover.jpg')
# cut book by the duration time of ar_source
book = book[:ar_source.shape[0], :, :, :]
# plt.imshow(ar_source[0])
# plt.show()

# print("Stop")
# 
#Write script for Q3.1
out = cv2.VideoWriter('../data/result/ar.avi', cv2.VideoWriter_fourcc('F','M','P','4'), 30, (book.shape[2],book.shape[1]))
for frame in range(book.shape[0]): 
    print("Image: " + str(frame))
    #crop ar_source 
    book_ratio = cv_cover.shape[0]/cv_cover.shape[1]
    # ar_width = int(ar_source.shape[2]/3) : int(2 * ar_source.shape[2]/3)
    ar_height = ar_source.shape[1]
    ar_width = ar_height / book_ratio
    book_middle = int(book.shape[2]/2)
    
    # ar_crop = ar_source[frame, :, int(ar_source.shape[2]/3) : int(2 * ar_source.shape[2]/3), :]
    ar_crop = ar_source[frame, :, int(book_middle - ar_width/2):int(book_middle + ar_width/2), :]
    # plt.imshow(ar_crop)
    # plt.show()
    # print("stop")

    # print("stop")
    #resize the cropped image into the size of cv_cover
    ar_resize = cv2.resize(ar_crop, dsize=(cv_cover.shape[1], cv_cover.shape[0]))
    # print("stop")
    #find homography between cv_cover and the book
    matches, locs1, locs2 = matchPics(cv_cover, book[frame], opts)
    x1 = locs1[matches[:, 0]]
    x2 = locs2[matches[:, 1]]
    x1[:, [1,0]] = x1[:, [0,1]]
    x2[:, [1,0]] = x2[:, [0,1]]

    bestH2to1, inliers = computeH_ransac(x1, x2, opts)
    
    # ar_shifted = cv2.warpPerspective(ar_resize, bestH2to1, (book.shape[2], book.shape[1]), flags=cv2.WARP_INVERSE_MAP)

    # plt.imshow(ar_shifted)
    # plt.show()
    # print("stop")

    composite_img = compositeH(bestH2to1, ar_resize, book[frame])
    # cv2.imshow("kungfupanda", composite_img.astype(np.uint8))
    # cv2.waitKey(0)
    
    out.write(composite_img)
out.release()