from __future__ import print_function

import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu3')

import os
import theano
import theano.tensor as T
import lasagne
import load_data as ld
import numpy as np
import lasagne.updates
import itertools
import time
import pandas as pd
import datetime
import matplotlib
matplotlib.use("Agg")

NAME = "BLSTM"
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
BATCH_SIZE = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 100
# Momentum
MOMENTUM = 0.9
DROPOUT = 0.5
INPUT_DROPOUT = 0.2
# Forget gate bias init
CONST_FORGET_B = 1.

# Path to HAR data
if 'nt' in os.name:
    ROOT_FOLDER = 'D:/PhD/Data/activity/'
else:
    ROOT_FOLDER = '/home/sdka/data/activity'


# Expand the target to all time steps
def expand_target(y, length):
    # return np.rollaxis(np.tile(y, (length, 1, 1)), 1,)
    return y


def load_data():
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = ld.LoadHAR(ROOT_FOLDER).uci_har_v1(add_pitch=True, add_roll=True)

    return dict(
        output_dim=int(y_test.shape[-1].eval()),
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        X_valid=x_valid,
        y_valid=y_valid,
        num_examples_train=x_train.shape[0].eval(),
        num_examples_valid=x_test.shape[0].eval(),
        num_examples_test=x_test.shape[0].eval(),
        seq_len=int(x_train.shape[1].eval()),
        n_fea=int(x_train.shape[2].eval())
        )


def build_model(output_dim, batch_size=BATCH_SIZE, seq_len=None, n_features=None):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, seq_len, n_features)
    )

    # Adding noise to weights can help training
    # l = layers.GaussianNoiseLayer(l_in)

    # Input dropout for regularization
    # l = lasagne.layers.dropout(l_in, p=INPUT_DROPOUT)

    # I'm using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    l_forward = lasagne.layers.LSTMLayer(
        l_in,
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
        l_in,
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

    # New BLSTM layer
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

    # New BLSTM layer
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

    # Slicing out the last units for classification
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

    loss_train = cross_ent_cost(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=False),
        y_batch
    )

    test_prediction = lasagne.layers.get_output(output_layer, X_batch, deterministic=True)
    loss_eval = cross_ent_cost(
        test_prediction,
        y_batch
    )

    accuracy = T.mean(
        T.eq(T.argmax(test_prediction, axis=1), T.argmax(y_batch, axis=1)),
        dtype=theano.config.floatX
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
        seq_len=dataset['seq_len'],
        n_features=dataset['n_fea']
        )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        )

    print("Starting training...")
    results = []
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset):
            print("{}: epoch {} of {} took {:.3f}s | training loss: {:.6f} "
                  "| validation loss: {:.6f} | validation accuracy: {:.2f} %".
                  format(NAME, epoch['number'], num_epochs, time.time() - now, epoch['train_loss'],
                         epoch['valid_loss'], epoch['valid_accuracy'] * 100))
            now = time.time()
            results.append([epoch['train_loss'], epoch['valid_loss'], epoch['valid_accuracy']])

            if epoch['train_loss'] is np.nan:
                break
            if epoch['number'] >= num_epochs:
                break
        # Save figure
        ax = pd.DataFrame(np.asarray(results), columns=['Training loss', 'Validation loss', 'Validation accuracy'])\
            .plot()
        fig = ax.get_figure()
        fig.savefig("%s-%s" % (NAME, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

    except KeyboardInterrupt:
        pass

    return output_layer

if __name__ == '__main__':
    main()
