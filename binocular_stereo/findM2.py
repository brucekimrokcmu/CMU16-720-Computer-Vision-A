
import numpy as np
import submission as sub
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
from helper import *
'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
pts1 = np.load("../data/some_corresp/pts1.npy")
pts2 = np.load("../data/some_corresp/pts2.npy")
I1 = plt.imread("../data/im1.png")
I2 = plt.imread("../data/im2.png")
M = np.maximum(I1.shape[1], I1.shape[0])
F = sub.eightpoint(pts1, pts2, M)

K1 = np.load("../data/intrinsics/K1.npy")
K2 = np.load("../data/intrinsics/K2.npy")
E = sub.essentialMatrix(F, K1, K2)
M2s = camera2(E)
identity = np.identity(3)
zeros = np.zeros((3,1))
M1 = np.hstack((identity,zeros))
C1 = K1 @ M1

M2_1 = M2s[:,:,0]
M2_2 = M2s[:,:,1]
M2_3 = M2s[:,:,2]
M2_4 = M2s[:,:,3]

C2_1 = K2 @ M2_1
C2_2 = K2 @ M2_2
C2_3 = K2 @ M2_3
C2_4 = K2 @ M2_4

C2_best = np.empty((3,4))

[w, err] = sub.triangulate(C1, pts1, C2_1, pts2)
if all(w[:,2]>0):
    C2_best = C2_1
    print("the best matrix is C2_1")
    np.savez("../data/q3_3.npz", C2_1)
[w, err] = sub.triangulate(C1, pts1, C2_2, pts2)
if all(w[:,2]>0):
    C2_best = C2_2
    print("the best matrix is C2_2")
    np.savez("../data/q3_3.npz", C2_2)
[w, err] = sub.triangulate(C1, pts1, C2_3, pts2)
if all(w[:,2]>0):
    C2_best = C2_3
    print("the best matrix is C2_3")
    np.savez("../data/q3_3.npz", C2_3)
[w, err] = sub.triangulate(C1, pts1, C2_4, pts2)
if all(w[:,2]>0):
    C2_best = C2_4
    print("the best matrix is C2_4")
    np.savez("../data/q3_3.npz", C2_4)
