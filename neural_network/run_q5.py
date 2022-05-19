import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt 

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
initialize_weights(1024,32,params,'layer1')
initialize_weights(32,32,params,'layer2')
initialize_weights(32,32,params,'layer3')
initialize_weights(32,1024,params,'output')

#initialize momentumsparams['m_boutput']
params['m_Wlayer1'] = np.zeros(params['Wlayer1'].shape)
params['m_blayer1'] = np.zeros(params['blayer1'].shape)
params['m_Wlayer2'] = np.zeros(params['Wlayer2'].shape)
params['m_blayer2'] = np.zeros(params['blayer2'].shape)
params['m_Wlayer3'] = np.zeros(params['Wlayer3'].shape)
params['m_blayer3'] = np.zeros(params['blayer3'].shape)
params['m_Woutput'] = np.zeros(params['Woutput'].shape)
params['m_boutput'] = np.zeros(params['boutput'].shape)
##########################

# should look like your previous training loops
total_loss_list = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        hl1 = forward(xb,params,name='layer1',activation=relu)
        hl2 = forward(hl1,params,name='layer2',activation=relu)
        hl3 = forward(hl2,params,name='layer3',activation=relu)
        probs = forward(hl3,params,name='output',activation=sigmoid) 
        loss = np.sum((probs - xb)**2)
        total_loss += loss
        delta1 = -2*(xb-probs)
        delta2 = backwards(delta1,params,name='output',activation_deriv=sigmoid_deriv)
        delta3 = backwards(delta2,params,name='layer3',activation_deriv=relu_deriv)
        delta4 = backwards(delta3,params,name='layer2',activation_deriv=relu_deriv)
        backwards(delta4,params,name='layer1',activation_deriv=relu_deriv)
        # apply gradient
        params['m_Wlayer1'] = 0.9*params['m_Wlayer1'] - params['grad_Wlayer1']*learning_rate
        params['m_blayer1'] = 0.9*params['m_blayer1'] - params['grad_blayer1']*learning_rate
        params['m_Wlayer2'] = 0.9*params['m_Wlayer2'] - params['grad_Wlayer2']*learning_rate
        params['m_blayer2'] = 0.9*params['m_blayer2'] - params['grad_blayer2']*learning_rate
        params['m_Wlayer3'] = 0.9*params['m_Wlayer3'] - params['grad_Wlayer3']*learning_rate
        params['m_blayer3'] = 0.9*params['m_blayer3'] - params['grad_blayer3']*learning_rate
        params['m_Woutput'] = 0.9*params['m_Woutput'] - params['grad_Woutput']*learning_rate
        params['m_boutput'] = 0.9*params['m_boutput'] - params['grad_boutput']*learning_rate
        ##########################
        params['Wlayer1'] += params['m_Wlayer1']
        params['blayer1'] += params['m_blayer1']
        params['Wlayer2'] += params['m_Wlayer2']
        params['blayer2'] += params['m_blayer2']
        params['Wlayer3'] += params['m_Wlayer3']
        params['blayer3'] += params['m_blayer3']
        params['Woutput'] += params['m_Woutput']
        params['boutput'] += params['m_boutput']

    total_loss /= train_x.shape[0]
    total_loss_list.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

epoch = np.arange(max_iters)
##### your code here #####
plt.plot(epoch, total_loss_list, label='total_loss')  # Plot some data on the (implicit) axes.
plt.xlabel('epoch')
plt.ylabel('total_loss')
plt.legend()
plt.show()

# Q5.3.1
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# visualize some results
##########################
##### your code here #####
index = [0, 50, 100, 150, 200]
im = np.empty((len(index), valid_x.shape[1]))
for i in range(len(index)):
    im[i,:] = valid_x[index[i],:]
    
hl1 = forward(im[0,:],params,name='layer1',activation=relu)
hl2 = forward(hl1,params,name='layer2',activation=relu)
hl3 = forward(hl2,params,name='layer3',activation=relu)
probs0 = forward(hl3,params,name='output',activation=sigmoid) 


hl1 = forward(im[2,:],params,name='layer1',activation=relu)
hl2 = forward(hl1,params,name='layer2',activation=relu)
hl3 = forward(hl2,params,name='layer3',activation=relu)
probs2 = forward(hl3,params,name='output',activation=sigmoid) 

plt.imshow(np.transpose(im[0,:].reshape(32,32)))
plt.imshow(np.transpose(probs0.reshape(32,32)))
plt.imshow(np.transpose(im[2,:].reshape(32,32)))
plt.imshow(np.transpose(probs2.reshape(32,32)))
plt.show()
##########################


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
hl1 = forward(valid_x,params,name='layer1',activation=relu)
hl2 = forward(hl1,params,name='layer2',activation=relu)
hl3 = forward(hl2,params,name='layer3',activation=relu)
probs_val = forward(hl3,params,name='output',activation=sigmoid) 

avg_PSNR = 0
for i in range(valid_x.shape[0]):
    PSNR = peak_signal_noise_ratio(valid_x[i], probs_val[i])
    avg_PSNR += PSNR 
    print(PSNR)
avg_PSNR = avg_PSNR/valid_x.shape[0]
print(avg_PSNR)
##########################
