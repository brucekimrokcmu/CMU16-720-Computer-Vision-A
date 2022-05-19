import numpy as np
from numpy.lib.function_base import i0
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # temp_patch # size(y,x) = (36,87) (167, 51)

    Ty_size = 36 # height - row
    Tx_size = 87 # width - col
    
    Ity = np.arange(0, It.shape[0]) #row
    Itx = np.arange(0, It.shape[1]) #col
    It_spline = RectBivariateSpline(Ity, Itx, It) #row, col, img
   
    Ty = np.linspace(rect[1], rect[3], Ty_size) 
    Tx = np.linspace(rect[0], rect[2], Tx_size)
    Txy = np.meshgrid(Ty, Tx) #row, col
    temp_patch = (It_spline.ev(Txy[0], Txy[1]))
    # Find It1_spline
    It1y = np.arange(0, It1.shape[0])
    It1x = np.arange(0, It1.shape[1])                                       
    It1_spline = RectBivariateSpline(It1y, It1x, It1)
    
    p = (np.copy(p0)).reshape((2,1))
    num = 0
    # update p
    while num < num_iters: 
        # Warp It1 with W(x;p) to get I(W(x;p))      
        Wy = np.linspace(rect[1]+p[1,0], rect[3]+p[1,0], Ty_size)  #row
        Wx = np.linspace(rect[0]+p[0,0], rect[2]+p[0,0], Tx_size)  #col  
        Wxy = np.meshgrid(Wy, Wx)    #row, col
        warp_patch = It1_spline.ev(Wxy[0], Wxy[1]) 

        # Compute error image
        error = -(warp_patch - temp_patch)
        
        # Warp gradient of I to compute delta
        dIt1dx = It1_spline.ev(Wxy[0], Wxy[1], dy =1)
        dIt1dy = It1_spline.ev(Wxy[0], Wxy[1], dx =1)
        dIt1dx_f = np.ndarray.flatten(dIt1dx)
        dIt1dy_f = np.ndarray.flatten(dIt1dy)
        grad_It1 = (np.vstack((dIt1dx_f, dIt1dy_f))).T
        # Evaluate Jacobian dW/dp
        jacob = np.array([[1, 0], [0, 1]])
        # Evaluate Hessian        
        A = np.dot(grad_It1, jacob)
        H = np.dot(A.T, A)     
        inv_H = np.linalg.inv(H)

        # Compute dp 
        dp = np.dot(inv_H, np.dot(A.T, error.flatten()).reshape((2,1)))
        p = p + dp
        if (np.linalg.norm(dp) < threshold):           
            break
        num += 1
       
    return p.T

      # # Compute Hessian
        # H = np.array([[0, 0], [0, 0]])
        # for y in range(warp_patch.s patches.Rectangle(hape[0]):
        #     for x in range(warp_patch.shape[1]):
        #         dIt1dx_t = dIt1dx[y][x]
        #         dIt1dy_t = dIt1dy[y][x]
        #         grad_It1 = np.array([[dIt1dx_t, dIt1dy_t]])
        #         H = H + np.transpose(grad_It1 @ jacob) @ (grad_It1 @ jacob)
            # dp = np.zeros(2)
        # for y in range(warp_patch.shape[0]):
        #     for x in range(warp_patch.shape[1]):
        #         dIt1dx_t = dIt1dx[y][x]
        #         dIt1dy_t = dIt1dy[y][x]
        #         grad_It1 = np.array([[dIt1dx_t, dIt1dy_t]])
        #         dp += (inv_H @ np.transpose(grad_It1 @ jacob) * (-error[y, x]))[:,0]