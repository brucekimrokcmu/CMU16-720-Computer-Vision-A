from matplotlib.image import composite_images
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.spatial.distance
import scipy.linalg
from matchPics import matchPics
from opts import get_opts

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
        
     #the below function requires x1 and x2 to be in order of matching pairs 
    A = np.empty((x1.shape[0]*2,9))
    # for i in np.random.randint(0, x1.shape[0], 4):
    for i in range(x1.shape[0]):
        # [x2, y2]  = H @ [x1, y1]
        # a = np.array([x1[i][0], x1[i][1], 1, 0, 0, 0, -x1[i][0]*x2[i][0], -x2[i][0]*x1[i][1], -x2[i][0]])
        # b = np.array([0, 0, 0, x1[i][0], x1[i][1], 1, -x1[i][0]*x2[i][1], -x1[i][1]*x2[i][1], -x2[i][1]])
        
        # [x1, y1]  = H @ [x2, y2]
        a = np.array([x2[i][0], x2[i][1], 1, 0, 0, 0, -x2[i][0]*x1[i][0], -x2[i][1]*x1[i][0], -x1[i][0]])
        b = np.array([0, 0, 0, x2[i][0], x2[i][1], 1, -x2[i][0]*x1[i][1], -x2[i][1]*x1[i][1], -x1[i][1]])
        A[2*i,:] = a
        A[2*i + 1,:] = b
        #A.append(a)
        #A.append(b)        
    #A = np.vstack(np.array(A))
    # print("stop")
    u, sig, h_temp = scipy.linalg.svd(A)
    h = h_temp[-1, :]
    H2to1 = h.reshape((3,3))
    H2to1 = H2to1/H2to1[2][2]
    # H2to1 = np.linalg.inv(H2to1)
    return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
    origin = np.array([0, 0])
    cent_x1 = np.mean(x1, axis=0)
    cent_x2 = np.mean(x2, axis=0)
    trans_x1 = origin - cent_x1
    trans_x2 = origin - cent_x2
    
	#Shift the origin of the points to the centroid
    T1 = np.array([[1, 0, trans_x1[0]], [0, 1, trans_x1[1]], [0, 0, 1]])
    T2 = np.array([[1, 0, trans_x2[0]], [0, 1, trans_x2[1]], [0, 0, 1]])

    ones_x1 = np.ones(x1.shape[0])
    aug_x1 = np.vstack((x1.T, ones_x1))

    ones_x2 = np.ones(x2.shape[0])
    aug_x2 = np.vstack((x2.T, ones_x2))

    shift_x1 = np.matmul(T1, aug_x1)
    shift_x2 = np.matmul(T2, aug_x2)
	
    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    
    scale_x1 = np.sqrt(2)/ np.amax(np.sqrt(np.square(shift_x1[0] - 0) + np.square(shift_x1[1] - 0)))
    scale_x2 = np.sqrt(2)/ np.amax(np.sqrt(np.square(shift_x2[0] - 0) + np.square(shift_x2[1] - 0)))
    
    T1[:2,:] *= scale_x1
    T2[:2,:] *= scale_x2

	#Similarity transform 1
    x1_norm = np.matmul(T1, aug_x1)
    x1_norm = np.delete(x1_norm, 2, 0)
    x1_norm = x1_norm.T

	#Similarity transform 2
    x2_norm = np.matmul(T2, aug_x2)
    x2_norm = np.delete(x2_norm, 2, 0)
    x2_norm = x2_norm.T
	
    #Compute homography
    H_norm = computeH(x1_norm, x2_norm)

	#Denormalization

    H2to1 = np.matmul(np.linalg.inv(T1), np.matmul(H_norm, T2)) #gives a calculated x1
    H2to1 = H2to1 / H2to1[2][2]
    # print("Stop")
    return H2to1



def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    ##again, assume that locs1 and locs2 are in the order of matching pair
    MAXONES = 0 
    for n in range(max_iters):
        ONECOUNT = 0
        x1 = []
        x2 = []
        random = np.arange(locs1.shape[0])
        for i in np.random.choice(random, size=4):
            x1.append(locs1[i])
            x2.append(locs2[i])   
        x1 = np.vstack(np.array(x1))
        x2 = np.vstack(np.array(x2))

        # H2to1 = computeH_norm(x1, x2)
        H2to1 = computeH(x1, x2)
        # ones_locs1 = np.ones(locs1.shape[0])
        # aug_locs1 = np.vstack((locs1.T, ones_locs1))

        ones_locs2 = np.ones(locs2.shape[0])
        aug_locs2 = np.vstack((locs2.T, ones_locs2))
        aug_locs1_cal = H2to1 @ aug_locs2
        aug_locs1_cal = np.divide(aug_locs1_cal, aug_locs1_cal[2, :], where=aug_locs1_cal[2, :] != 0)
        locs1_cal = aug_locs1_cal[:2, :].T


        # dist = np.linalg.norm(locs1_cal - locs1, axis =1)
        dist = np.sqrt(np.square(locs1[:, 0] - locs1_cal[:, 0]) + np.square(locs1[:, 1] - locs1_cal[:, 1]))
        # print("stop")     
        for i, element in enumerate(dist):
            if element <= inlier_tol:
                dist[i] = 1
            else:
                dist[i] = 0
        inliers = dist       
        for element in inliers:
            if element == 1:
                ONECOUNT += 1
        if ONECOUNT > MAXONES:
            MAXONES = ONECOUNT
            bestH2to1 = H2to1
    return bestH2to1, inliers



def compositeH(H2to1, template, img):

    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.

    # inv_H2to1 = np.linalg.inv(H2to1) 
    
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         for k in range(img.shape[2]):
    #             if img[i, j, k] != 0:
    #                 template[i, j, k] = img[i, j, k]
    

    # plt.imshow(template)
    # plt.show()

    # print("stop")

    #Create mask of same size as template

    #Warp mask by appropriate homography

    #Warp template by appropriate homography

    #Use mask to combine the warped template and the image

    mask = np.zeros([template.shape[0], template.shape[1],3], dtype=np.uint8)
    mask.fill(255) 
    mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP) 
    # print("stop")
    # mask = np.where(mask !=0, template, img)
    
    template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    # print("stop")    
    OUTPUT = np.where(mask != 0, template, img)
    OUTPUT = np.flip(OUTPUT, axis=2)
    # OUTPUT = OUTPUT[:, :, [2,1,0]]
   
    composite_img = OUTPUT

    return composite_img


