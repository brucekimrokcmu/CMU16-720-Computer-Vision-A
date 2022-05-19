# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import tifffile as tiff
import skimage.color
def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    l = light/(np.sqrt(light[0]**2 + light[1]**2 + light[2]**2))  
    #meshgrid xy to get z
    x = np.arange(-res[0]/2, res[0]/2)
    y = np.arange(-res[1]/2, res[1]/2)
    x = pxSize*x
    y = pxSize*y
    xv, yv = np.meshgrid(x, y)
    mesh_size = xv.size
    z = np.sqrt(rad**2 - xv**2 - yv**2) 

    xv = xv.reshape(mesh_size, 1)
    yv = yv.reshape(mesh_size, 1)
    z  = z.reshape(mesh_size,1)
    #find normal vectors
    normal = np.hstack((xv,yv,z))/rad
    #get image
    image = (normal @ np.transpose(l)).reshape(res[1],res[0]) 

    return image


    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
def loadData(path = "../data/"):    

    I = None
    for i in range(7):
        #read image and convert rgb to xyz
        src = cv2.imread(path + "input_{}.tif".format(i+1))
        img = cv2.cvtColor(src, cv2.COLOR_BGR2XYZ)
        if I is None:
            h, w, _ = img.shape
            size = h*w
            I = np.empty((7,size))      
        #retrive y channel 
        I[i,:] = img[:,:,1].reshape(1, size)
    #get L and s
    L = np.load(os.path.join(path,"sources.npy")).T   
    s = (h, w)

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    B = np.linalg.inv(L@L.T)@L@I
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    albedos = np.sqrt(np.sum(B**2, axis=0))
    normals = B/albedos
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
        
    albedoIm = albedos.reshape(s)
    normalIm = (normals.T).reshape(s[0],s[1],3)

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    zx = np.reshape(-normals[0, :]/(normals[2, :]), s)
    zy = np.reshape(-normals[1, :]/(normals[2, :]), s)
    surface = integrateFrankot(zx, zy)

    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface '2' as a data type >>> e

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    h, w = surface.shape
    y, x = np.arange(h), np.arange(w)
    fig = plt.figure()
    xv, yv = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(xv, yv, surface, cmap='coolwarm')
    plt.show()


if __name__ == '__main__':

    #setting parameters
    rad = 75e-4
    pxSize = 7e-6
    res = np.array((3840,2160))
    center = np.array((0,0,0))
    light = np.array((1,1,1))
    # light = np.array((1,-1,1))
    # light = np.array((-1,-1,1))

    img = renderNDotLSphere(center, rad, light, pxSize, res)
    plt.imshow(img, cmap='gray')
    plt.show()

    I, L, s = loadData(path = "../data/")
    u, sing, v = np.linalg.svd(I, full_matrices=False)

    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)    
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    #plot albedos
    plt.imshow(albedoIm, cmap = "gray")
    plt.show()
    #plot normalIm
    normalIm = (normalIm - np.min(normalIm))/(np.max(normalIm) - np.min(normalIm))
    plt.imshow(normalIm, cmap = "rainbow")
    plt.show()
    #plot 3d surface
    surface = estimateShape(normals, s)
    plotSurface(surface)

    pass
