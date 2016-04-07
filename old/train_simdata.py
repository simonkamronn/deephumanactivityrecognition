import numpy as np 
import theano
import theano.tensor as T

import lasagne
from lasagne import nonlinearities
from lasagne.theano_extensions import conv
import lasagne.updates
import scipy.io as sio
import time

tobool = lambda x: np.asarray(x, dtype=np.bool)


# DATASET_PATH = "/home/sedielem/data/urbansound8k/spectrograms.h5"
MINIBATCHSIZE = 32
LEARNING_RATE = 0.01
#MOMENTUM = 0.9
#WEIGHT_DECAY = 0.0

NCONV1 = 20
NCONV2 = 20
NHID = 50
DROPOUT = 0.5
INPUTDROPOUT=0.2
N_EPOCHS = 500
# SOFTMAX_LAMBDA = 0.01

DATAFILE='data/simdata.mat'
trainsplits = [1,2,3]
valsplits = [4]
testsplits = [5]


#Xtrain = NSAMPLES X NFEATURES X SEQLEN
#Xval = NSAMPLES X NFEATURES X SEQLEN
#Xtest = NSAMPLES X NFEATURES X SEQLEN

#ytrain = NSAMPLES X N_CLASSES
#yval = NSAMPLES X N_CLASSES
#ytest = NSAMPLES X N_CLASSES



#LOAD DATA
data = sio.loadmat(DATAFILE)

input = list()
input.append(np.atleast_3d(data['input1']).astype(theano.config.floatX))
input.append(np.atleast_3d(data['input2']).astype(theano.config.floatX))
input.append(np.atleast_3d(data['input3']).astype(theano.config.floatX))
input.append(np.atleast_3d(data['input4']).astype(theano.config.floatX))
input.append(np.atleast_3d(data['input5']).astype(theano.config.floatX))

target = list()
target.append(tobool(data['target1'].astype(theano.config.floatX)))
target.append(tobool(data['target2'].astype(theano.config.floatX)))
target.append(tobool(data['target3'].astype(theano.config.floatX)))
target.append(tobool(data['target4'].astype(theano.config.floatX)))
target.append(tobool(data['target5'].astype(theano.config.floatX)))


train_ind = [int(s)-1 for s in trainsplits]
val_ind = [int(s)-1 for s in valsplits]
test_ind = [int(s)-1 for s in testsplits]

Xtrain = np.concatenate([input[i] for i in train_ind], axis=0)
ytrain = np.concatenate([target[i] for i in train_ind], axis=0)

Xval = np.concatenate([input[i] for i in val_ind], axis=0)
yval = np.concatenate([target[i] for i in val_ind], axis=0)

Xtest = np.concatenate([input[i] for i in test_ind], axis=0)
ytest = np.concatenate([target[i] for i in test_ind], axis=0)


N_FEATURES = Xtrain.shape[1]
SEQLEN = Xtrain.shape[2]
N_CLASSES = ytrain.shape[1]


#Set up symbolic and shared variables
input = T.tensor3('input')
target_output = T.matrix('target_output')

batch_size_X = (MINIBATCHSIZE, N_FEATURES, SEQLEN)
batch_size_y = (MINIBATCHSIZE, N_CLASSES)
sh_input = theano.shared(np.zeros(shape=batch_size_X, dtype=theano.config.floatX), borrow=True)
sh_target_output = theano.shared(np.zeros(shape=batch_size_y, dtype=theano.config.floatX), borrow=True)

givens_train = [(input, sh_input), (target_output, sh_target_output)]
givens_test = [(input, sh_input)]

#construct the network using lasagne layers

#TODO: IM NOT SHURE WHICH CONVOLUTION TO USE i.e. conv1d_unstrided, conv1d_mc0 etc?

l = lasagne.layers.InputLayer((MINIBATCHSIZE, N_FEATURES, SEQLEN))
l = lasagne.layers.dropout(l, p=INPUTDROPOUT)
#conv layers hould take input of (BATCH_SIZE, NFEAT, SEQLEN) i think
l = lasagne.layers.Conv1DLayer(l, num_filters=NCONV1, filter_length=3, stride=1, convolution=conv.conv1d_mc0, nonlinearity=nonlinearities.rectify)
l = lasagne.layers.Conv1DLayer(l, num_filters=NCONV1, filter_length=3, stride=1, convolution=conv.conv1d_mc0, nonlinearity=nonlinearities.rectify)
l = lasagne.layers.dropout(l, p=DROPOUT)
#l = lasagne.layers.Conv1DLayer(l, num_filters=NCONV2, filter_length=3, stride=2, convolution=conv.conv1d_mc1, nonlinearity=nonlinearities.rectify)
#l = lasagne.layers.dropout(l, p=DROPOUT)
l = lasagne.layers.DenseLayer(l, num_units=NHID, nonlinearity=nonlinearities.rectify)
l = lasagne.layers.dropout(l, p=DROPOUT)
l = lasagne.layers.DenseLayer(l, num_units=N_CLASSES, nonlinearity=T.nnet.softmax)

#get all params in the network (used to calculated the gradient in the update function (sgd, adagrad)
all_params = lasagne.layers.get_all_params(l)
param_count = sum([np.prod(p.get_value().shape) for p in all_params])
print "parameter count: %d" % param_count

#Creates a costfunction
def costfun(ypred, ytar):
    #Assumes both are inputs are encoded as one-hot encodings
    #ypred: MINIBATCHSIZE x N_CLASSES (one hot encoded)
    #ytar: MINIBATCHSIZE x N_CLASSES (one hot encoded)

    #Convert ytar to class label
    true_class = T.argmax(ytar,axis=1)

    total_error = -T.log(ypred[T.arange(MINIBATCHSIZE), true_class] + 1e-8)
    cross_ent = T.sum(total_error)/MINIBATCHSIZE
    return cross_ent


cost_train = costfun(l.get_output(input, deterministic=False), target_output)
cost_val = costfun(l.get_output(input, deterministic=True), target_output)

print 'Computing Updates...'
#Various gradient descent algorithms - nesterovs seems to work well
#Note that the results seems quite sensitive to the Learning rate -> if the code doesn't converge try to lower the LR
#updates = lasagne.updates.adagrad(cost_train, all_params, learning_rate=LEARNING_RATE, epsilon=1e-6 )
#updates = lasagne.updates.momentum(cost_train, all_params, learning_rate=LEARNING_RATE)
#updates = lasagne.updates.momentum(cost_train, all_params, learning_rate=1, epsilon=1e-6 )
updates = lasagne.updates.nesterov_momentum(cost_train, all_params, learning_rate=LEARNING_RATE, momentum=0.9)




print 'Computing Functions...'
train = theano.function([], cost_train, updates=updates, givens=givens_train,on_unused_input='warn')
compute_cost_val = theano.function([], cost_val, givens=givens_train, on_unused_input='warn') #maybe this is not nescessary
compute_preds = theano.function([],l.get_output(input, deterministic=True),givens=givens_test, on_unused_input='warn')


def calcPerformance(X,y):

    N_SAMPLES = X.shape[0]
    N_BATCH = int(np.ceil(float(N_SAMPLES) / MINIBATCHSIZE))
    N_SAMPLES_PAD = N_BATCH*MINIBATCHSIZE

    samples = range(N_SAMPLES_PAD) #no need to shuffle for performance calculation
    #wraps the indexes around so they never exceed N_SAMPLES_TRAIN i.e. ensures that the sample ids are not out of bounds
    samples_mod = [x % N_SAMPLES for x in samples]
    batches=[samples_mod[(i)*MINIBATCHSIZE:(i+1)*MINIBATCHSIZE] for i in range(N_BATCH)]


    yclass_true = np.zeros((N_SAMPLES_PAD),dtype=np.int32)
    yclass_pred = np.zeros((N_SAMPLES_PAD),dtype=np.int32)
    for j,ind in enumerate(batches):

        # update shared variables
        y_batch = y[ind,]
        X_batch = X[ind,]
        sh_target_output.set_value(y_batch)
        sh_input.set_value(X_batch)

        #compute y_preds using X_batch as input
        y_preds = compute_preds()

        #store true and predicted class labels
        yclass_true[ind] = np.argmax(y_batch,axis=1)
        yclass_pred[ind] = np.argmax(y_preds,axis=1)


    #Calculated accuracy
    #remove samples that were "out of bounds" i.e. wrapped around. If not doing this the val/test results might be slightly biased
    # i.e if N_SAMPLES_TEST = 5, and MINIBATCHSIZE = 3 -> N_BATCH_TEST = 2 and N_SAMPLES_TEST_PAD = 6
    # => BATCH1 = [0,1,2], BATCH2 = [3,4,0]. Hence we need to account for calculating the performance of some samples twice.
    idkeep = np.array(samples) < N_SAMPLES
    yclass_true = yclass_true[idkeep]
    yclass_pred = yclass_pred[idkeep]

    Acc = np.sum(yclass_true == yclass_pred)/float(len(yclass_true))
    return Acc, yclass_pred, yclass_true




print 'Training...'


#Calculate the train batch sizes
N_SAMPLES_TRAIN = Xtrain.shape[0]
N_BATCH_TRAIN = int(np.ceil(float(N_SAMPLES_TRAIN) / MINIBATCHSIZE))
N_SAMPLES_TRAIN_PAD = N_BATCH_TRAIN*MINIBATCHSIZE

Acc_train = []
Acc_val = []
Acc_test = []
for epoch in range(N_EPOCHS):
    #START TRAINING

    #shuffle the training sample
    shuffle_samples = np.random.permutation(N_SAMPLES_TRAIN_PAD)
    #wraps the indexes around so they never exceed N_SAMPLES_TRAIN i.e. ensures that the sample ids are not out of bounds
    shuffle_samples = [x % N_SAMPLES_TRAIN for x in shuffle_samples]
    batches_train=[shuffle_samples[(i)*MINIBATCHSIZE:(i+1)*MINIBATCHSIZE] for i in range(N_BATCH_TRAIN)]

    start_time = time.time()
    c = 0
    print "Mini-batches Done: "
    for j,ind in enumerate(batches_train):
        c += 1
        if c % 5 == 0:
            print '%i ' % c,

        # update shared variables
        y_batch = ytrain[ind,]
        X_batch = Xtrain[ind,]

        sh_target_output.set_value(y_batch)
        sh_input.set_value(X_batch)

        #training and updting parameters for a single minibatch
        train()

    #END TRAIN

    #Train Performance
    acc_train, yclass_pred_train, yclass_true_train = calcPerformance(Xtrain,ytrain)
    Acc_train.append(acc_train)


    #VALIDATION / TEST PERFORMANCE
    acc_val, yclass_pred_val, yclass_true_val = calcPerformance(Xval,yval)
    Acc_val.append(acc_val)

    acc_test, yclass_pred_test, yclass_true_test = calcPerformance(Xtest,ytest)
    Acc_test.append(acc_test)

    end_time = time.time()

    print "EPOCH %i DONE using %f seconds" % (epoch, end_time-start_time)
    print "Acc train: %0.3f, Acc val: %0.3f, Acc test: %0.3f" % (acc_train, acc_val, acc_test)
