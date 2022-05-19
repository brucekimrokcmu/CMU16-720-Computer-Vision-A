import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanadeAffine import LucasKanadeAffine

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

frames = seq.shape[2]

for frame in range(frames-1):
    It = seq[:,:, frame]
    It1 = seq[:,:, frame+1]  

    M = LucasKanadeAffine(It, It1, threshold, num_iters)
    fig, ax = plt.subplots()
    ax.imshow(It1, cmap = 'gray')