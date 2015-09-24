from __future__ import print_function

import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu2')

import os
import theano
import theano.tensor as T
import lasagne
import load_data as ld
import numpy as np
import lasagne.updates
from lasagne.objectives import categorical_crossentropy
from lasagne.layers import LSTMLayer, Gate, InputLayer, DenseLayer, ConcatLayer, SliceLayer, get_output
import itertools
import time
import pandas as pd
import datetime
import matplotlib
matplotlib.use("Agg")

NAME ="CNN_BLSTM"
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 200
# Number of training sequences in each batch
BATCH_SIZE = 50
# Number of features
N_FEATURES = 5
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 5
# Number of epochs to train the net
NUM_EPOCHS = 500
# Momentum
MOMENTUM = 0.9
DROPOUT = 0.5
INPUT_DROPOUT = 0.2
# Forget gate init bias
CONST_FORGET_B = 0.

# Path to HAR data
if 'nt' in os.name:
    ROOT_FOLDER = 'D:/PhD/Data/activity/'
else:
    ROOT_FOLDER = '/home/sdka/data/activity'


def load_data():
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid), (sequence_length, n_features, n_classes) \
        = ld.LoadHAR(ROOT_FOLDER).uci_har_v1(add_pitch=False, add_roll=False)

    return dict(
        output_dim=n_classes,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        X_valid=x_valid,
        y_valid=y_valid,
        num_examples_train=x_train.shape[0].eval(),
        num_examples_valid=x_test.shape[0].eval(),
        num_examples_test=x_test.shape[0].eval(),
        seq_len=sequence_length,
        n_fea=n_features
        )


def build_model(output_dim, batch_size=BATCH_SIZE, seq_len=None, n_features=N_FEATURES):
    x = np.random.random((batch_size, seq_len, n_features)).astype(theano.config.floatX)
    sym_x = T.tensor3('x')

    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, seq_len, n_features)
    )
    print("Input shape", get_output(l_in, sym_x).eval({sym_x: x}).shape)

    # Input dropout for regularization
    # l_in = lasagne.layers.dropout(l_in, p=INPUT_DROPOUT)

    # Subsample signal with averaging
    l_sub_samp = lasagne.layers.FeaturePoolLayer(l_in, pool_size=2, pool_function=T.mean)
    seq_len /= 2
    l_reshp = lasagne.layers.ReshapeLayer(l_sub_samp, (batch_size, 1, seq_len, n_features))
    print("Conv2D temporal feature shape", get_output(l_sub_samp, sym_x).eval({sym_x: x}).shape)

    # Convolutional layers
    l_conv = lasagne.layers.Conv2DLayer(l_reshp, num_filters=10, filter_size=(3, 1), pad=(1, 0))
    print("Conv output shape", get_output(l_conv, sym_x).eval({sym_x: x}).shape)

    # Reshape layer back to normal structure
    l_dim = lasagne.layers.DimshuffleLayer(l_conv, (0, 2, 1, 3))
    l_reshp = lasagne.layers.ReshapeLayer(l_dim, (batch_size, seq_len, -1))
    print("BLSTM input shape", get_output(l_reshp, sym_x).eval({sym_x: x}).shape)

    l_prev = l_reshp
    num_layers = 3
    for _ in range(num_layers):
        l_forward = LSTMLayer(
            l_prev,
            num_units=N_HIDDEN/2,
            ingate=Gate(
                W_in=lasagne.init.HeUniform(),
                W_hid=lasagne.init.HeUniform()
            ),
            grad_clipping=GRAD_CLIP,
            forgetgate=Gate(
                b=lasagne.init.Constant(CONST_FORGET_B)
            ),
            nonlinearity=lasagne.nonlinearities.tanh
        )
        l_backward = LSTMLayer(
            l_prev,
            num_units=N_HIDDEN/2,
            ingate=Gate(
                W_in=lasagne.init.HeUniform(),
                W_hid=lasagne.init.HeUniform()
            ),
            grad_clipping=GRAD_CLIP,
            forgetgate=Gate(
                b=lasagne.init.Constant(CONST_FORGET_B)
            ),
            nonlinearity=lasagne.nonlinearities.tanh,
            backwards=True
        )
        print("LSTM forward shape", get_output(l_forward, sym_x).eval({sym_x: x}).shape)

        l_prev = ConcatLayer(
            [l_forward, l_backward],
            axis=2
        )
        print("LSTM concat shape", get_output(l_prev, sym_x).eval({sym_x: x}).shape)

    # Slicing out the last units for classification
    l_forward_slice = SliceLayer(l_forward, -1, 1)
    l_backward_slice = SliceLayer(l_backward, 0, 1)

    print("LSTM forward slice shape", get_output(l_forward_slice, sym_x).eval({sym_x: x}).shape)
    l_cat2 = ConcatLayer(
        [l_forward_slice, l_backward_slice],
        axis=1
    )

    # Our output layer is a simple dense connection, with n_classes output unit
    print("Dense input shape", get_output(l_cat2, sym_x).eval({sym_x: x}).shape)
    l_out = DenseLayer(
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
    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)

    prediction = T.argmax(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
        axis=1
    )
    accuracy = T.mean(T.eq(prediction, T.argmax(y_batch, axis=1)), dtype=theano.config.floatX)

    epsilon = 1e-8
    loss_train = categorical_crossentropy(
        T.clip(
            lasagne.layers.get_output(output_layer, X_batch, deterministic=False),
            epsilon, 1-epsilon),
        y_batch
    ).mean()

    loss_eval = categorical_crossentropy(
        T.clip(
            lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
            epsilon, 1-epsilon),
        y_batch
    ).mean()

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.rmsprop(
        loss_train,
        all_params,
        learning_rate=0.0001,
        rho=0.8
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

    results = []
    print("Starting training...")
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
    output_layer = main()

