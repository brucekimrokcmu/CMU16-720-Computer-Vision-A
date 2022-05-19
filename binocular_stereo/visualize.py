'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import submission as sub
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
from helper import *

pts1 = np.load("../data/some_corresp/pts1.npy")
pts2 = np.load("../data/some_corresp/pts2.npy")
I1 = plt.imread("../data/im1.png")
I2 = plt.imread("../data/im2.png")
M = np.maximum(I1.shape[1], I1.shape[0])
F = sub.eightpoint(pts1, pts2, M)
#displayEpipolarF(I1, I2, F)

K1 = np.load("../data/intrinsics/K1.npy")
K2 = np.load("../data/intrinsics/K2.npy")
E = sub.essentialMatrix(F, K1, K2)
M2s = camera2(E)
M2_3 = M2s[:,:,2]


identity = np.identity(3)
zeros = np.zeros((3,1))
M1 = np.hstack((identity,zeros))
C1 = K1 @ M1
C2_3 = K2 @ M2_3

[w, err] = sub.triangulate(C1, pts1, C2_3, pts2)

wx = w[:,0]
wy = w[:,1]
wz = w[:,2]

x1 = np.load("../data/templeCoords/x1.npy")
y1 = np.load("../data/templeCoords/y1.npy")

[x2, y2] = sub.epipolarCorrespondence(I1, I2, F, x1, y1)

pts1_4 = np.hstack((x1,y1))
pts2_4 = np.hstack((x2,y2))



np.savez("../data/q4_2.npz", F, M1, M2_3, C1, C2_3)

[w, err] = sub.triangulate(C1, pts1_4, C2_3, pts2_4)

wx = w[:,0]
wy = w[:,1]
wz = w[:,2]

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim3d(-0.6 , 0.6)
ax.set_ylim3d(np.min(wy) , np.max(wy))
ax.set_zlim3d(3, 5)
ax.scatter(wx, wy, wz)
plt.show()

# print("stop")