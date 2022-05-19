import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade 
from PIL import Image

# write your script here, we recommend the above libraries for making your animation
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
rect_orignal = np.copy(rect)
frames = seq.shape[2]
carseqrects = []

psum = np.empty((1,2))
i = 0
for frame in range(frames-1):
    # set initial values
    It = seq[:,:, frame]
    It1 = seq[:,:, frame+1]    
    # get p to update rect 
    # print("stop")
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))  
    psum += p
    
    
    fig, ax = plt.subplots()
    ax.imshow(It1, cmap = 'gray')
    box = patches.Rectangle((rect[0], rect[3]), rect[2] - rect[0], rect[1] - rect[3], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(box)
    # fig.savefig(("../data/frame "+ str(i) + ".png"))
    plt.close(fig)
    print("image: "+str(i))
    i += 1
    rect = [rect_orignal[0]+ psum[0,0], rect_orignal[1]+psum[0,1], rect_orignal[2]+psum[0,0], rect_orignal[3]+psum[0,1]]
    carseqrects.append(rect)


carseqrects = np.vstack(np.array(carseqrects))
np.save("../data/carseqrects.npy", carseqrects)

for frameNum in [1, 100, 200, 300, 400]:
    rect = carseqrects[frameNum]

    fig,ax = plt.subplots(1)
    ax.imshow(seq[:,:,frameNum], cmap = 'gray')

    box = patches.Rectangle((rect[0], rect[3]), rect[2] - rect[0], rect[1] - rect[3], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(box)
    plt.axis('off')

    plt.savefig("../data/car {}.png".format(frameNum), pad_inches=0, bbox_inches='tight', transparent=True)
