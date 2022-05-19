from PIL.Image import Image
import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt 

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = None
learning_rate = None
hidden_size = 64    
##########################
##### your code here #####
batch_size = 5
learning_rate = 1e-3
# learning_rate = 1e-2
# learning_rate = 1e-4
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

# valid_batches = get_random_batches(valid_x,valid_y,batch_size)
# valid_batch_num = len(valid_batches)

params = {}

# initialize layers here
##########################
##### your code here #####
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')
##########################
import copy
params_orig = copy.deepcopy(params)
train_acc = []
train_loss = []
valid_acc = []
valid_loss = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    valid_total_loss = 0
    valid_avg_acc = 0
    batch_count = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        # forward
        hl1 = forward(xb, params, name='layer1', activation=sigmoid)
        probs = forward(hl1, params, name='output', activation=softmax) 
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        avg_acc = (acc + avg_acc*batch_count)/(batch_count+1)
        batch_count += 1
        # backward
        delta1 = probs-yb
        delta2 = backwards(delta1,params,name='output',activation_deriv=linear_deriv)
        backwards(delta2,params,name='layer1',activation_deriv=sigmoid_deriv)
        # apply gradient
        params['Wlayer1'] -= params['grad_Wlayer1']*learning_rate
        params['blayer1'] -= params['grad_blayer1']*learning_rate
        params['Woutput'] -= params['grad_Woutput']*learning_rate
        params['boutput'] -= params['grad_boutput']*learning_rate
        ##########################
    train_acc.append(avg_acc)
    train_loss.append(total_loss/train_x.shape[0])
    #run the network on validation
    batch_count = 0

# for xb, yb in valid_batches:
    hl1 = forward(valid_x, params, name='layer1', activation=sigmoid)
    probs = forward(hl1, params, name='output', activation=softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_total_loss += loss
    valid_avg_acc = (acc + valid_avg_acc*batch_count)/(batch_count+1)
    batch_count += 1
    valid_acc.append(valid_avg_acc)
    valid_loss.append(valid_total_loss/valid_x.shape[0])

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,valid_total_loss,valid_avg_acc))

# run on validation set and report accuracy! should be above 75%

##########################
final_valid_acc = valid_avg_acc
epoch = np.arange(max_iters)
##### your code here #####
plt.plot(epoch, train_acc, label='train_acc')  # Plot some data on the (implicit) axes.
plt.plot(epoch, valid_acc, label='valid_acc')  # etc.
plt.xlabel('epoch')
plt.ylabel('accuaracy')
plt.legend()
plt.show()
##########################

plt.plot(epoch, train_loss, label='train_loss')  # Plot some data on the (implicit) axes.
plt.plot(epoch, valid_loss, label='valid_loss')  # etc.
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

print('Validation accuracy: ',final_valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
fig = plt.figure(figsize=(4,4))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
for i, img in zip(grid, np.transpose(params_orig['Wlayer1'])):
    vis_weight = img.reshape(32,32)
    i.imshow(vis_weight)


plt.show()

fig = plt.figure(figsize=(4,4))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
for i, img in zip(grid, np.transpose(params['Wlayer1'])):
    vis_weight = img.reshape(32,32)
    i.imshow(vis_weight)


plt.show()

##########################

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))


test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
# compute comfusion matrix here
##########################
##### your code here #####
#test data x will have classified to - and what they should be 
# neural_network_output(test_x) vs test_y
hl1 = forward(test_x, params, name='layer1', activation=sigmoid)
probs = forward(hl1, params, name='output', activation=softmax)

pred = np.argmax(probs, axis=1)
ans = np.argmax(test_y, axis=1)

for i in range(test_y.shape[0]):
    confusion_matrix[pred[i],ans[i]] += 1

##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()