import numpy as np
from numpy.lib.function_base import _percentile_dispatcher
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    d = np.sqrt(6/(in_size + out_size))
    W = np.random.uniform(-d, d, (in_size, out_size))
    b = np.zeros((out_size))
    ##########################

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    z = np.exp(-x)
    res = 1 / (1 + z)
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    ##### your code here #####
    pre_act = X @ W + b
    post_act = activation(pre_act)
    ##########################


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    x_shift = (x - np.expand_dims(np.max(x, axis=1), axis=1))
    x_exp = np.exp(x_shift) 
    res = x_exp / np.expand_dims(np.sum(x_exp, axis=1), axis=1)
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    loss = -np.sum(y * np.log(probs))
    
    predict = np.argmax(probs, axis=1)
    ans = np.argmax(y, axis=1) 
    acc = np.count_nonzero(ans==predict)/len(ans)

    ##########################

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    delta_new = delta*activation_deriv(post_act)
    grad_X = delta_new @ W.T
    grad_W = X.T @ delta_new
    grad_b = np.sum(delta_new, axis=0)

    ##########################

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    randind = index[: len(index) - len(index)%batch_size] 
    
    for i in range(0, x.shape[0]//batch_size):
        allocate_index = randind[i*batch_size:(i+1)*batch_size]
        batch_x = x[allocate_index]
        batch_y = y[allocate_index]
        batches.append((batch_x, batch_y))
    
    randindlast = index[len(index) - len(index)%batch_size :]
    for i in randindlast:
        allocate_index = randindlast[(i+1)*batch_size]
        batch_x = x[allocate_index]
        batch_y = y[allocate_index]
        batches.append((batch_x, batch_y))
    
    ##########################
    return batches
