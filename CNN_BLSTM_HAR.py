from __future__ import print_function

import theano
import theano.tensor as T
import lasagne
import load_data as ld
import numpy as np
import lasagne.updates
import itertools
import time
import pandas as pd
#import matplotlib.pyplot as plt

# Number of input units (window samples)
N_UNITS = 128
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 200
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
NUM_EPOCHS = 500
# Momentum
MOMENTUM = 0.9
DROPOUT = 0.5
INPUT_DROPOUT = 0.2
# Forget gate init bias
CONST_FORGET_B = 1.

# Path to HAR data
# ROOT_FOLDER = 'D:/PhD/Data/activity/'
ROOT_FOLDER = '/home/sdka/data/activity'

# Expand the target to all time steps
def expand_target(y, length):
    return y


def load_data():
    data = ld.LoadHAR(ROOT_FOLDER).uci_har_v1()

    return dict(
        output_dim=int(data['y_test'].shape[-1]),
        X_train=theano.shared(data['x_train'].astype(theano.config.floatX)),
        y_train=theano.shared(
            expand_target(data['y_train'], N_UNITS).astype(theano.config.floatX)
        ),
        X_valid=theano.shared(data['x_test'].astype(theano.config.floatX)),
        y_valid=theano.shared(
            expand_target(data['y_test'], N_UNITS).astype(theano.config.floatX)
        ),
        X_test=theano.shared(data['x_test'].astype(theano.config.floatX)),
        y_test=theano.shared(
            expand_target(data['y_test'], N_UNITS).astype(theano.config.floatX)
        ),
        num_examples_train=data['x_train'].shape[0],
        num_examples_valid=data['x_test'].shape[0],
        num_examples_test=data['x_test'].shape[0],
        seq_len=int(data['x_train'].shape[1]),
        n_fea=int(data['x_train'].shape[2])
        )


def build_model(output_dim, batch_size=BATCH_SIZE, seq_len=None):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, seq_len, N_FEATURES)
    )

    # Input dropout for regularization
    l = lasagne.layers.dropout(l_in, p=INPUT_DROPOUT)

    # Reshape layer for convolution
    l = lasagne.layers.DimshuffleLayer(l, (0, 2, 1))

    # Pad to keep original sequence length. Each conv layers takes 2 samples
    l_pad = lasagne.layers.PadLayer(l, 2)

    # Convolution layers. Convolution, pooling and dropout
    l_conv = lasagne.layers.Conv1DLayer(l_pad,  num_filters=24, filter_size=3)
    l_pool = lasagne.layers.MaxPool1DLayer(l_conv, pool_size=2)
    l_conv = lasagne.layers.Conv1DLayer(l_pool, num_filters=48, filter_size=3)
    l_pool = lasagne.layers.MaxPool1DLayer(l_conv, pool_size=2)
    l_conv = lasagne.layers.Conv1DLayer(l_pool, num_filters=96, filter_size=3)
    l_conv = lasagne.layers.Conv1DLayer(l_conv, num_filters=96, filter_size=3)
    l_pool = lasagne.layers.MaxPool1DLayer(l_conv, pool_size=2)

    # Reshape layer back to normal
    l = lasagne.layers.DimshuffleLayer(l, (0, 2, 1))

    # BLSTM layers
    l_forward = lasagne.layers.LSTMLayer(
        l,
        num_units=N_HIDDEN/2,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.HeUniform(),
            W_hid=lasagne.init.HeUniform()
        ),
        grad_clipping=GRAD_CLIP,
        forgetgate=lasagne.layers.Gate(
            b=lasagne.init.Constant(CONST_FORGET_B)
        )
    )
    l_backward = lasagne.layers.LSTMLayer(
        l,
        num_units=N_HIDDEN/2,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.HeUniform(),
            W_hid=lasagne.init.HeUniform()
        ),
        grad_clipping=GRAD_CLIP,
        forgetgate=lasagne.layers.Gate(
            b=lasagne.init.Constant(CONST_FORGET_B)
        ),
        backwards=True
    )

    l_cat = lasagne.layers.ConcatLayer(
        [l_forward, l_backward],
        axis=2
    )

    l_forward = lasagne.layers.LSTMLayer(
        l_cat,
        num_units=N_HIDDEN/2,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.HeUniform(),
            W_hid=lasagne.init.HeUniform()
        ),
        grad_clipping=GRAD_CLIP,
        forgetgate=lasagne.layers.Gate(
            b=lasagne.init.Constant(CONST_FORGET_B)
        )
    )
    l_backward = lasagne.layers.LSTMLayer(
        l_cat,
        num_units=N_HIDDEN/2,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.HeUniform(),
            W_hid=lasagne.init.HeUniform()
        ),
        grad_clipping=GRAD_CLIP,
        forgetgate=lasagne.layers.Gate(
            b=lasagne.init.Constant(CONST_FORGET_B)
        ),
        backwards=True
    )

    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)
    l_backward_slice = lasagne.layers.SliceLayer(l_backward, 0, 1)

    l_cat2 = lasagne.layers.ConcatLayer(
        [l_forward_slice, l_backward_slice],
        axis=1
    )

    # Our output layer is a simple dense connection, with n_classes output unit
    l_out = lasagne.layers.DenseLayer(
        l_cat2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax
    )

    return l_out


def cross_ent_cost(predicted_values, target_values):
    return -T.sum(target_values*T.log(predicted_values + 1e-8))/T.sum(target_values)


def create_iter_functions(dataset, output_layer,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE,
                          momentum=MOMENTUM):
    """
    Create functions for training, validation and testing to iterate one epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = T.tensor3('input')
    y_batch = T.matrix('target_output')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    prediction = T.argmax(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
        axis=1
    )
    accuracy = T.mean(T.eq(prediction, T.argmax(y_batch, axis=1)), dtype=theano.config.floatX)

    loss_train = cross_ent_cost(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=False),
        y_batch
    )

    loss_eval = cross_ent_cost(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
        y_batch
    )

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.rmsprop(
        loss_train,
        all_params,
        learning_rate=0.002,
        rho=0.95
    )

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
    )

def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }

def main(num_epochs=NUM_EPOCHS):
    print("Loading data...")
    dataset = load_data()

    print("Building model and compiling functions...")
    output_layer = build_model(
        output_dim=dataset['output_dim'],
        batch_size=BATCH_SIZE,
        seq_len=dataset['seq_len']
        )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        )

    results = []
    print("Starting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))
            results.append([epoch['train_loss'], epoch['valid_loss'], epoch['valid_accuracy']])

            if epoch['train_loss'] is np.nan:
                break
            if epoch['number'] >= num_epochs:
                break
        df = pd.DataFrame(np.asarray(results), columns=['Training loss', 'Validation loss', 'Validation accuracy'])
    except KeyboardInterrupt:
        pass

    return output_layer, df

if __name__ == '__main__':
    output_layer, df = main()
    #df.plot()
    #plt.show()
