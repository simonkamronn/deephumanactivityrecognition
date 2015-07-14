import lasagne
import theano
import theano.tensor as T
import numpy as np
import time
import load_data as ld
import logging

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('experiment.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# Number of input units (window samples)
N_UNITS = 128
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
BATCH_SIZE = 100
# Number of features
N_FEATURES = 3
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10
# Momentum
MOMENTUM = 0.9
# Path to HAR data
ROOT_FOLDER = 'D:/PhD/Data/activity/'

def make_batches(X, length, batch_size=BATCH_SIZE):
    '''
    Convert a list of matrices into batches of uniform length
    :parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size
    :returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Mask denoting whether to include each time step of each time series
            matrix
    '''
    n_batches = len(X)//batch_size
    if X.ndim > 2:
        X_batch = np.zeros((n_batches, batch_size, length, X.shape[-1]),
                           dtype=theano.config.floatX)
    else:
        X_batch = np.zeros((n_batches, batch_size, X.shape[-1]),
                           dtype=theano.config.floatX)

    X_mask = np.zeros(X_batch.shape, dtype=np.bool)
    for b in range(n_batches):
        for n in range(batch_size):
            X_m = X[b*batch_size + n]
            X_batch[b, n, :X_m.shape[0]] = X_m[:length]
            X_mask[b, n, :X_m.shape[0]] = 1
    return X_batch, X_mask

logger.info('Loading data...')
data = ld.LoadHAR(ROOT_FOLDER).uci_har_v1()

# Find the longest sequence
length = int(max(max([X.shape[0] for X in data['x_train']]),
                 max([X.shape[0] for X in data['x_test']])))

# Expand the target to all time steps
def expand_target(y, length):
    return np.rollaxis(np.tile(y, (length, 1, 1)), 1,)

# Convert to batches of time series of uniform length
X_train, _ = make_batches(data['x_train'], length)
y_train, train_mask = make_batches(expand_target(data['y_train'], length), length)
X_val, _ = make_batches(data['x_test'], length)
y_val, val_mask = make_batches(expand_target(data['y_test'], length), length)

n_epochs = 500
learning_rate = 10
momentum = .9

l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, length, X_val.shape[-1]))

l_forward = lasagne.layers.LSTMLayer(l_in, num_units=N_HIDDEN)
l_backward = lasagne.layers.LSTMLayer(l_in, num_units=N_HIDDEN)
l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])

l_reshape = lasagne.layers.ReshapeLayer(l_sum, (BATCH_SIZE*length, N_HIDDEN))

nonlinearity = lasagne.nonlinearities.tanh
l_rec_out = lasagne.layers.DenseLayer(l_reshape,
                                      num_units=y_val.shape[-1],
                                      nonlinearity=nonlinearity)

l_out = lasagne.layers.ReshapeLayer(l_rec_out,
                                    (BATCH_SIZE, length, int(y_val.shape[-1])))

# Cost function is mean squared error
input = T.tensor3('input')
target_output = T.tensor3('target_output')
mask = T.tensor3('mask')

def cost(output):
    return -T.sum(mask*target_output*T.log(output))/T.sum(mask)

cost_train = cost(l_out.get_output(input, deterministic=False))
cost_eval = cost(l_out.get_output(input, deterministic=True))


# Use SGD for training
all_params = lasagne.layers.get_all_params(l_out)
logger.info('Computing updates...')
updates = lasagne.updates.momentum(cost_train, all_params,
                                   learning_rate, momentum)
logger.info('Compiling functions...')
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output, mask], cost_train,
                        updates=updates)
y_pred = theano.function([input], l_out.get_output(input, deterministic=True))
compute_cost = theano.function([input, target_output, mask], cost_eval)

logger.info('Training...')
# Train the net
for epoch in range(n_epochs):
    start_time = time.time()
    batch_shuffle = np.random.choice(X_train.shape[0], X_train.shape[0], False)
    for sequences, labels, sequence_mask in zip(X_train[batch_shuffle],
                                                y_train[batch_shuffle],
                                                train_mask[batch_shuffle]):
        sequence_shuffle = np.random.choice(sequences.shape[0],
                                            sequences.shape[0], False)
        train(sequences[sequence_shuffle], labels[sequence_shuffle],
              sequence_mask[sequence_shuffle])
    end_time = time.time()
    cost_val = sum([compute_cost(X_val_n, y_val_n, mask_n)
                    for X_val_n, y_val_n, mask_n,
                    in zip(X_val, y_val, val_mask)])

    y_val_pred = np.array([y_pred(X_val_n) for X_val_n in X_val])
    y_val_labels = np.argmax(y_val*val_mask, axis=-1).flatten()
    y_val_pred_labels = np.argmax(y_val_pred*val_mask, axis=-1).flatten()
    n_time_steps = np.sum(val_mask)/val_mask.shape[-1]
    error = np.sum(y_val_labels != y_val_pred_labels)/float(n_time_steps)
    logger.info("Epoch {} took {}, cost = {}, error = {}".format(
        epoch, end_time - start_time, cost_val, error))