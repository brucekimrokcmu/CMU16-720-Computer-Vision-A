# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimatePseudonormalsCalibrated
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    #finding svd 
    u, s, v = np.linalg.svd(I, full_matrices=False)
    
    #top k to zeros    
    s[3:] = 0
    B = v[:3, :]
    L = u[:3, :]

    return B, L


if __name__ == "__main__":

    # Put your main code here
    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    #mu nu lamda that works -> flat -> other plays
    mu = 1 #-1.0, -1, 1, 1.5
    nu = 0.5 #0.1, 0.5, 1, 1.5
    lamda = -1. #2.5, -1, 1, 2.5
    G = np.asarray([[1, 0, 0], [0, 1, 0], [mu, nu, lamda]])
    B = np.linalg.inv(G.T)@(B)

    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    #plot albedos
    plt.imshow(albedoIm, cmap = "gray")
    plt.show()

    #normalize normalIm - plot normalIm
    normalIm = (normalIm - np.min(normalIm))/(np.max(normalIm) - np.min(normalIm))
    plt.imshow(normalIm, cmap = "rainbow")
    plt.show()
    
    #plot 3D surface
    surface = estimateShape(normals, s)
    plotSurface(surface)

    pass
