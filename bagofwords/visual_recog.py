import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K

    hist = np.histogram(wordmap, bins=K, range=(0,K), density=True)
    #print(hist)
    # ----- TODO -----
    return hist[0]

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.nd

    ## exarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L

    hist_cat = []
    
    #how to define l with respect to L? 
    for l in range(L):
        cat_cell_hist = []
        split0 = wordmap.shape[0] // 2**l
        split1 = wordmap.shape[1] // 2**l
        for cell_index in range(0, 2**l):
            cell_wordmap = wordmap[cell_index * split0 : cell_index * split0 + split0, cell_index * split1 : cell_index * split1 + split1]
            cell_hist = get_feature_from_wordmap(opts, cell_wordmap)
            cat_cell_hist.append(cell_hist)
            
        normalized_cell_hist = np.ndarray.flatten(np.array(cat_cell_hist)) / (4**l)
        if l < 2:
            weighted_cell_hist = normalized_cell_hist * (1/2**(L))
        else:
            weighted_cell_hist = normalized_cell_hist * (1/2**(L+1-l)) 
        #hist_cat = np.ndarray.flatten(weighted_cell_hist) 
        hist_cat.append(weighted_cell_hist)
    #print(type(cell_hist))
    #print(type(hist_cat))    
    hist_all = np.ndarray.flatten(np.array(hist_cat))
    #print(hist_all.size)
    #print(hist_all.shape)
    # ----- TODO -----
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K) <--this should be corrected/ hist_all
    '''
    data_dir = opts.data_dir
    K = opts.K
    img = Image.open(data_dir + "/" + img_path)
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    # ----- TODO -----
    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    #loop through every image and call get_img_feature
    feature_list = []
    for track, train_path in enumerate(train_files):
        print("Image", track + 1)
        feature = get_image_feature(opts, train_path, dictionary)
        feature_list.append(feature)
    features = np.vstack(feature_list)

        
    # ----- TODO -----

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
        
    #np.subtract with(word_hist, histograms)
    #for each row, find the maximum value and store it to a column vector
    #distance = 1 - column vector.. might have to expand the dimension of matrix with all elements '1' 
    
    # max_difference = np.amax(np.absolute(np.subtract(word_hist, histograms)),1)
    # similarity = max_difference.reshape(-1, 1)
    # ones = np.ones_like(similarity)
    # hist_dist = np.subtract(ones,similarity)

    distance = np.sum(np.minimum(word_hist, histograms), axis = 1)
    hist_dist = 1 - distance

    # ----- TODO -----
    return hist_dist  
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)

    conf = np.zeros((8, 8))
        
    for count, test_path in enumerate(test_files):
        print("Image count", count + 1)
        img = Image.open(data_dir + "/" + test_path)
        test_wordmap = visual_words.get_visual_words(test_opts, img, dictionary)
        test_wordhist = get_feature_from_wordmap_SPM(test_opts, test_wordmap)
        distance = distance_to_set(test_wordhist, features)
        predicted_label = train_labels[np.argmin(distance)]
        conf[test_labels[count]][predicted_label] += 1
    

    numer = 0
    for i in range(8):
        numer += conf[i, i]
    denom = np.sum(conf)
    accuracy = 100*numer/denom

    # ----- TODO -----
    return conf, accuracy

    # train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    # for train_path in train_files[:10]:
    #     img = Image.open(data_dir + "/" + train_path)
    #     test_wordmap = visual_words.get_visual_words(test_opts, img, dictionary)
    #     test_wordhist = get_feature_from_wordmap_SPM(test_opts, test_wordmap)
    #     distance = distance_to_set(test_wordhist, features)
    #     predicted_label = train_labels[np.argmin(distance)]
    #     conf[test_labels[count]][predicted_label] += 1
    #     count += 1
    