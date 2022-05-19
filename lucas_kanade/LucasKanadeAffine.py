import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

# def getAffineJac(x, y):
    # return np.array([[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]]

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return M: the Affine warp matrix [2x3 numpy array] 
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.array([M[0,0] - 1, M[0,1], M[0,2], M[1,0], M[1,1] - 1, M[1,2]]).reshape(6,1)

    Ity = np.arange(0, It.shape[0]) #row
    Itx = np.arange(0, It.shape[1]) #col
    It_spline = RectBivariateSpline(Ity, Itx, It) #row, col, img    
    Itxy = np.meshgrid(Ity, Itx)


    It1y = np.arange(0, It1.shape[0])
    It1x = np.arange(0, It1.shape[1])                                       
    It1xy = np.meshgrid(It1y, It1x)
    It1_spline = RectBivariateSpline(It1y, It1x, It1)
    It1xy_arr = np.array(It1xy)

    num = 0
    while num < num_iters: 
        
        y_coord = np.array([np.dot(It1xy_arr[1,:,:], p[3,0]) + np.dot(It1xy_arr[0,:,:], (1+p[4,0])) + p[5,0]]) #y
        x_coord = np.array([np.dot(It1xy_arr[1,:,:], (1+p[0,0])) + np.dot(It1xy_arr[0,:,:], p[1,0]) + p[2,0]]) #x

        y_legal = np.logical_and(y_coord >= 0, y_coord < It.shape[0])
        x_legal = np.logical_and(x_coord >= 0, x_coord < It.shape[1])
        legal_coord = np.logical_and(y_legal, x_legal)
        warped_It1 = It1_spline.ev(legal_coord, legal_coord)

        template = (It_spline.ev(legal_coord, legal_coord))

        # jacob_aff = np.zeros((It.shape[0], It.shape[1]))
        print("stop")
        error = -(warped_It1 - template)
        print("stop")
        
        
        num += 1
    return M


