import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.updates
import itertools
import time
import datetime
import pandas as pd

import utils
import load_data as ld
import os

import matplotlib
matplotlib.use("Agg")
# import matplotlib.pyplot as plt

BATCH_SIZE = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0

DROPOUT = 0.5
INPUTDROPOUT = 0.2
NUM_EPOCHS = 1000
# SOFTMAX_LAMBDA = 0.01

NAME = "2D_CNN"

# Path to HAR data
if 'nt' in os.name:
    ROOT_FOLDER = 'D:/PhD/Data/activity/'
else:
    ROOT_FOLDER = '/home/sdka/data/activity'



def load_data():
    data = ld.LoadHAR(ROOT_FOLDER).uci_har_v1()
    x_test = utils.spectrogram_2d(utils.magnitude(data['x_test'])).astype(theano.config.floatX)

    return dict(
        output_dim=int(data['y_test'].shape[-1]),
        X_train=theano.shared(utils.spectrogram_2d(utils.magnitude(data['x_train'])).astype(theano.config.floatX)),
        y_train=theano.shared(
            data['y_train'].astype(theano.config.floatX)
        ),
        X_valid=theano.shared(x_test),
        y_valid=theano.shared(
            data['y_test'].astype(theano.config.floatX)
        ),
        X_test=theano.shared(x_test),
        y_test=theano.shared(
            data['y_test'].astype(theano.config.floatX)
        ),
        num_examples_train=data['x_train'].shape[0],
        num_examples_valid=data['x_test'].shape[0],
        num_examples_test=data['x_test'].shape[0],
        seq_len=int(x_test.shape[1]),
        input_width=x_test.shape[2],
        input_height=x_test.shape[3]
        )


def build_model(input_width, input_height, output_dim, batch_size=BATCH_SIZE):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, input_width, input_height),
        )

    # l_in_dropout = lasagne.layers.DropoutLayer(l_in, p=INPUTDROPOUT)

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=24,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters=24,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    l_conv3 = lasagne.layers.Conv2DLayer(
        l_conv2,
        num_filters=48,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    l_conv4 = lasagne.layers.Conv2DLayer(
        l_conv3,
        num_filters=48,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    l_pool = lasagne.layers.Pool2DLayer(l_conv4, pool_size=(2, 2))

    x = np.random.random((batch_size, 1, input_width, input_height)).astype(theano.config.floatX)
    sym_x = T.tensor4('x')
    model = lasagne.layers.get_output(l_pool, sym_x)
    out = model.eval({sym_x: x})
    print("out shape", out.shape)

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform(),
        )

    return l_out


def cross_ent_cost(predicted_values, target_values):
    return -T.sum(target_values*T.log(predicted_values + 1e-8))/T.sum(target_values)


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.matrix('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    prediction = T.argmax(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
        axis=-1
    )
    accuracy = T.mean(T.eq(prediction, T.argmax(y_batch, axis=-1)), dtype=theano.config.floatX)

    loss_train = cross_ent_cost(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=False),
        y_batch
    )

    loss_eval = cross_ent_cost(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
        y_batch
    )

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.adam(
        loss_train,
        all_params,
        learning_rate=learning_rate)

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
        input_height=dataset['input_height'],
        input_width=dataset['input_width'],
        output_dim=dataset['output_dim'],
        batch_size=BATCH_SIZE
        )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        X_tensor_type=T.tensor4,
        )

    results = []
    print("Starting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset):
            print("{}: epoch {} of {} took {:.3f}s | training loss: {:.6f} "
                  "| validation loss: {:.6f} | validation accuracy: {:.2f} %%".
                  format(NAME, epoch['number'], num_epochs, time.time() - now, epoch['train_loss'],
                         epoch['valid_loss'], epoch['valid_accuracy'] * 100))

            results.append([epoch['train_loss'], epoch['valid_loss'], epoch['valid_accuracy']])

            now = time.time()
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
