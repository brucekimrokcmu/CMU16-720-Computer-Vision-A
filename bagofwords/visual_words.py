import os, multiprocessing
from os.path import join, isfile
from matplotlib.pyplot import sci

import numpy as np
from PIL import Image
import scipy.ndimage
from sklearn import cluster
import skimage.color
import random

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    '''
    1. Check if image is float, normalize  
    2. Check if RGB / Greyscale [no. of channels]
      2-1 if RGB --> LAB
      2-2 if greyscale --> duplicate to 3 channels and stack  
    3. Convolve each channel with filter & stack 
    4. Stack all channels 
    5. Apply all filters with 3 scales  
    '''
    #1. Check if image is float, normalize  
    if np.amax(img) > 1:
        img = np.array(img).astype(np.float32)/255
    #2.check if RGB or greyscale

    if len(np.shape(img)) < 3:
        dup_img = np.dstack((img, img, img))
        img = dup_img

    F = 4 * len(opts.filter_scales)

    img = skimage.color.rgb2lab(img)
    filter_response = np.empty((img.shape[0], img.shape[1], 3*F))
    filter_scales = opts.filter_scales
    #Gaussian
    count = 0
    for i in opts.filter_scales:
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_filter(img[:,:,j], i)
            filter_response[:,:,count] = img_convolve
            count += 1
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_laplace(img[:,:,j], i)
            filter_response[:,:,count] = img_convolve
            count += 1
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_filter(img[:,:,j], i, order = (0, 1))
            filter_response[:,:,count] = img_convolve
            count += 1
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_filter(img[:,:,j], i, order = (1, 0))
            filter_response[:,:,count] = img_convolve
            count += 1 
    

    # ----- TODO -----
    return filter_response

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    '''
    Complete compute dictionary by: 
    Read in each image in the set 
     in one image to visual_words func 
     Use the retunred stack of features and sample alpha random points 
     Add the sampled alpha random filter responses to the stack 
     Run k-means on the training set 
     Save the dictionary as the cluster centers 
    
   '''
    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker=1):
    ''' 
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha 
    F = 4 * len(opts.filter_scales)

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    #convovle each image first 
    #then extract a random pixel to the size of alpha 
    #stack them

    con_filter_responses = np.empty((1, 3*F))
    for track, train_path in enumerate(train_files):
        print("Image", track+1)
        train_img = Image.open(data_dir + "/" + train_path)
        if np.amax(train_img) > 1:
            train_img = np.array(train_img).astype(np.float32)/255
        
        if len(np.shape(train_img)) < 3:
            dup_img = np.dstack((train_img, train_img, train_img))
            train_img = dup_img

        train_img = skimage.color.rgb2lab(train_img)
    
        temp_train_filter_response = np.empty((train_img.shape[0], train_img.shape[1], 3*F))
        count = 0
        for i in opts.filter_scales:
            for j in range(3):
                train_img_convolve = scipy.ndimage.gaussian_filter(train_img[:,:,j], i)
                temp_train_filter_response[:,:,count] = train_img_convolve
                count += 1
            for j in range(3):
                train_img_convolve = scipy.ndimage.gaussian_laplace(train_img[:,:,j], i)
                temp_train_filter_response[:,:,count] = train_img_convolve
                count += 1
            for j in range(3):
                train_img_convolve = scipy.ndimage.gaussian_filter(train_img[:,:,j], i, order = [0,1])
                temp_train_filter_response[:,:,count] = train_img_convolve
                count += 1
            for j in range(3):
                train_img_convolve = scipy.ndimage.gaussian_filter(train_img[:,:,j], i, order = [1,0])
                temp_train_filter_response[:,:,count] = train_img_convolve
                count += 1
        #resize from H x W x 3F to HW X 3F
        resized_train_filter_response = temp_train_filter_response.reshape((temp_train_filter_response.shape[0]*temp_train_filter_response.shape[1], temp_train_filter_response.shape[2]))
        #then I need to extract 25(=size of alpha) random pixel from the resized filter response
        sample_indexes = np.random.random_integers(0, resized_train_filter_response.shape[0]-1, alpha)
        filter_responses = resized_train_filter_response[sample_indexes, :] 
        con_filter_responses = np.concatenate((con_filter_responses, filter_responses), axis=0)
        con_filter_responses = con_filter_responses[1:,:]

    kmeans = cluster.KMeans(n_clusters=K).fit(con_filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    # ----- TODO -----
    return dictionary

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    F = 4 * len(opts.filter_scales)
    filter_scales = opts.filter_scales
    K = opts.K
    #1. Check if image is float, normalize  
    if np.amax(img) > 1:
        img = np.array(img).astype(np.float32)/255
    #2.check if RGB or greyscale
    if len(np.shape(img)) < 3:
        dup_img = np.dstack((img, img, img))
        img = dup_img

    img = skimage.color.rgb2lab(img)
    #3. get a filter response of an image
    filter_response = np.empty((img.shape[0], img.shape[1], 3*F))
    count = 0
    for i in opts.filter_scales:
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_filter(img[:,:,j], i)
            filter_response[:,:,count] = img_convolve
            count += 1
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_laplace(img[:,:,j], i)
            filter_response[:,:,count] = img_convolve
            count += 1
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_filter(img[:,:,j], i, order = (0, 1))
            filter_response[:,:,count] = img_convolve
            count += 1
        for j in range(3):
            img_convolve = scipy.ndimage.gaussian_filter(img[:,:,j], i, order = (1, 0))
            filter_response[:,:,count] = img_convolve
            count += 1 
    #4. Resize of the filter response of image to HW X 3F
    dim1 = filter_response.shape[0]*filter_response.shape[1]
    dim2 = filter_response.shape[2]
    resized_filter_response = filter_response.reshape((dim1, dim2))
    
    distance = scipy.spatial.distance.cdist(resized_filter_response, dictionary)
    wordmap = np.zeros((resized_filter_response.shape[0], 1))
    for i in range(distance.shape[0]):
        wordmap[i] = np.argmin(distance[i,:])
    wordmap = wordmap.reshape((filter_response.shape[0],filter_response.shape[1]))
    # ----- TODO -----
    return wordmap
