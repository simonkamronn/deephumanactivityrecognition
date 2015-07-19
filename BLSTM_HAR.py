from __future__ import print_function

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as layers
import lasagne.layers.recurrent as recurrent
import load_data as ld
import numpy as np
import lasagne.updates
import itertools
import time


# Number of input units (window samples)
N_UNITS = 128
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 128
# Number of training sequences in each batch
BATCH_SIZE = 100
# Number of features
N_FEATURES = 3
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 5
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10
# Momentum
MOMENTUM = 0.9
DROPOUT = 0.5
INPUT_DROPOUT=0.2

# Path to HAR data
ROOT_FOLDER = 'D:/PhD/Data/activity/'

# Expand the target to all time steps
def expand_target(y, length):
    return np.rollaxis(np.tile(y, (length, 1, 1)), 1,)

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
    l_in = layers.InputLayer(
        shape=(batch_size, seq_len, N_FEATURES)
    )

    l = layers.GaussianNoiseLayer(l_in)

    # Input dropout for regularization
    l = layers.dropout(l, p=INPUT_DROPOUT)

    # I'm using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    l_forward = recurrent.LSTMLayer(
        l,
        num_units=N_HIDDEN,
        grad_clipping=GRAD_CLIP
    )
    l_backward = recurrent.LSTMLayer(
        l,
        num_units=N_HIDDEN,
        grad_clipping=GRAD_CLIP,
        backwards=True)

    # Now, we'll concatenate the outputs to combine them.
    # l_con = lasagne.layers.ConcatLayer(
    #     [l_forward, l_backward],
    #     axis=-1
    # )

    # Sum the layers
    l = layers.ElemwiseSumLayer([l_forward, l_backward])

    # In order to connect a recurrent layer to a dense layer, we need to
    # flatten the first two dimensions (our "sample dimensions"); this will
    # cause each time step of each sequence to be processed independently
    l = layers.ReshapeLayer(
        l,
        (-1, N_HIDDEN)
    )

    # Our output layer is a simple dense connection, with n_classes output unit
    l = layers.DenseLayer(
        l,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax
    )

    # To reshape back to our original shape
    l_out = layers.ReshapeLayer(
        l,
        (batch_size, seq_len, output_dim)
    )

    return l_out


def max_vote(m):
    m = T.argmax(m, axis=2)
    results = []
    for i in range(m.shape[0]):
        results[i] = T.argmax(T.bincount(m[i]))
    return results


def cross_ent_cost(predicted_values, target_values):
    # TODO: max vote sequences
    return -T.mean(T.sum(target_values*T.log(predicted_values + 1e-8), axis=target_values.ndim-1))


def create_iter_functions(dataset, output_layer,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE,
                          momentum=MOMENTUM):
    """
    Create functions for training, validation and testing to iterate one epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = T.tensor3('input')
    y_batch = T.tensor3('target_output')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    loss_train = cross_ent_cost(
        layers.get_output(output_layer, X_batch, deterministic=False),
        y_batch
    )

    test_prediction = layers.get_output(output_layer, X_batch, deterministic=True)
    loss_eval = cross_ent_cost(
        test_prediction,
        y_batch
    )

    accuracy = T.mean(
        T.eq(T.argmax(test_prediction, axis=2), T.argmax(y_batch, axis=2)),
        dtype=theano.config.floatX
    )

    all_params = layers.get_all_params(output_layer)
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

            if epoch['train_loss'] is np.nan:
                break
            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    return output_layer

if __name__ == '__main__':
    main()