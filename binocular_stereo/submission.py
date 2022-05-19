"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
from re import A
import numpy as np
from numpy.lib.shape_base import apply_along_axis
from util import refineF
import scipy.linalg
import scipy.ndimage
import cv2

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementatptsion
    U = np.empty((pts1.shape[0],9))  
    scale = [[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]]
    ones = np.ones((pts1.shape[0],1))
    aug_pts1 = np.hstack((pts1, ones))
    aug_pts2 = np.hstack((pts2, ones))
    
    norm_pts1 = scale @ aug_pts1.T
    norm_pts2 = scale @ aug_pts2.T

    for i in range(pts1.shape[0]):
        x1 = norm_pts1[0, i]
        y1 = norm_pts1[1, i]
        x2 = norm_pts2[0, i]
        y2 = norm_pts2[1, i]

        a = np.array([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])
        U[i, :] = a
    
    u, sig, f_temp = scipy.linalg.svd(U)
    f = f_temp[-1, :]
    norm_F = f.reshape((3,3))
    F = np.transpose(scale) @ norm_F @ scale

    F = refineF(F, pts1, pts2)
    
    np.savez("../data/q2_1.npz", F, M)

    return F



'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''

def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    
    E = K1.T @ F @ K2


    np.savez("../data/q3_1.npz", E)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    wpts = np.empty((pts1.shape[0],4))
    w = np.empty((pts1.shape[0],3))
    err1 = 0
    err2 = 0
    err = 0 

    for i in range(pts1.shape[0]):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]

        a = [C1[0,0]-C1[2,0]*x1, C1[0,1]-C1[2,1]*x1, C1[0,2]-C1[2,2]*x1, C1[0,3]-C1[2,3]*x1] 
        b = [-C1[1,0]+C1[2,0]*y1, -C1[1,1]+C1[2,1]*y1, -C1[1,2]+C1[2,2]*y1, -C1[1,3]+C1[2,3]*y1]
        c = [C2[0,0]-C2[2,0]*x2, C2[0,1]-C2[2,1]*x2, C2[0,2]-C2[2,2]*x2, C2[0,3]-C2[2,3]*x2]
        d = [-C2[1,0]+C2[2,0]*y2, -C2[1,1]+C2[2,1]*y2, -C2[1,2]+C2[2,2]*y2, -C2[1,3]+C2[2,3]*y2]

        A = np.vstack((a,b,c,d))
       
        u, sig, c_temp = scipy.linalg.svd(A)
        wpts[i] = c_temp[-1, :]      
        
        pts1_hat = C1 @ wpts[i]
        pts2_hat = C2 @ wpts[i]
        
        x1_hat = pts1_hat[0]/pts1_hat[2]
        y1_hat = pts1_hat[1]/pts1_hat[2]
        x2_hat = pts2_hat[0]/pts2_hat[2]
        y2_hat = pts2_hat[1]/pts2_hat[2]

        err1 += (pts1[i, 0] - x1_hat)**2 + (pts1[i, 1] - y1_hat)**2
        err2 += (pts2[i, 0] - x2_hat)**2 + (pts2[i, 1] - y2_hat)**2
        err = err1 + err2

        Px = wpts[i,0]/wpts[i,3]
        Py = wpts[i,1]/wpts[i,3]
        Pz = wpts[i,2]/wpts[i,3]

        w[i, 0] = Px
        w[i, 1] = Py
        w[i, 2] = Pz
        # print(A)
    # print("stop")
    return w, err
  
'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
'''

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    ones = np.ones((x1.shape[0],1))
    pts_im1 = np.transpose(np.hstack((x1, y1, ones)))
    temp =  F @ pts_im1
    A = temp[0,:]
    B = temp[1,:]
    C = temp[2,:]
    # eplin = A*x2 + B*y2 + C = 0
    im1g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    PATCH_SIZE = 40
    im2_y2 = np.arange(im2g.shape[0]+PATCH_SIZE)      
    pad_im2g = np.pad(im2g, PATCH_SIZE)
    x2 = np.empty((x1.shape[0],1))
    y2 = np.empty((y1.shape[0],1))

    L= PATCH_SIZE*2
    SIGMA = 5   
    ax = np.linspace(-(L - 1) / 2., (L - 1) / 2., L)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(SIGMA))
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel)

    
    for i in range(x1.shape[0]):
        x1min = np.asscalar(x1[i] - PATCH_SIZE)
        x1max = np.asscalar(x1[i] + PATCH_SIZE)
        y1min = np.asscalar(y1[i] - PATCH_SIZE)
        y1max = np.asscalar(y1[i] + PATCH_SIZE)
        im1g_patch = kernel*im1g[y1min:y1max, x1min:x1max]
        # print("stop")
        ERRORMIN = 999999999999999
        ref = np.asscalar(y1[i])
        for j in range(ref-50, ref+50):
            im2_x2 = (-B[i]*im2_y2[j+PATCH_SIZE] - C[i])/A[i]
            im2_x2 = (np.rint(im2_x2)).astype(int)
            x2min = np.asscalar(im2_x2 - PATCH_SIZE)
            x2max = np.asscalar(im2_x2 + PATCH_SIZE)
            y2min = np.asscalar(im2_y2[j+PATCH_SIZE] - PATCH_SIZE)
            y2max = np.asscalar(im2_y2[j+PATCH_SIZE] + PATCH_SIZE)
            im2g_patch = kernel*pad_im2g[y2min:y2max, x2min:x2max]
            error = np.sum((im1g_patch - im2g_patch)**2)
            # print(im2_x2)
            
            if error < ERRORMIN:
                ERRORMIN = error
                x2[i] = (im2_x2).astype(int)
                y2[i] = (im2_y2[j+PATCH_SIZE]).astype(int)
        # print(i)
        # print("x2: "+ str(x2[i]))
        # print("y2: "+ str(y2[i]))
                
        # print("stop")
    return x2, y2
'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your impplementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
