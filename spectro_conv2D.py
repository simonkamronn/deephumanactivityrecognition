import numpy as np 
import theano
import theano.tensor as T
import lasagne
from lasagne import nonlinearities
import lasagne.updates
import itertools
import scipy.io as sio
import time
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from matplotlib.mlab import specgram


BATCH_SIZE = 500
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
NUM_HIDDEN_UNITS = 512

DROPOUT = 0.5
INPUTDROPOUT = 0.2
NUM_EPOCHS = 1000
# SOFTMAX_LAMBDA = 0.01

# calculate the magnitude of the accelerometer
def magnitude(x_in):
    return np.sqrt((x_in*x_in).sum(axis=1))

def spectro_conv(X):
    """
    Convert array of epoched accelerometer time series to spectrograms
    :param X: accelerometer data of dim samples x channels x window length
    :return: spectrogram of dim samples x 24 x 24 where channels are concatenated
    """
    X = X[:, 5:8]  # Only use 3 acc channels
    N_BINS = 16
    N_WIN, N_FEA, N_SAMP = X.shape
    NFFT = 128
    noverlap = NFFT - N_SAMP/N_BINS
    # X = magnitude(np.swapaxes(X[:, 0:3, :], 1, 2).reshape(-1, 3))
    # X = 10. * np.log10(specgram(np.pad(X, pad_width=NFFT/2-1, mode='constant'), NFFT=NFFT, noverlap=noverlap)[0])
    # X = X.reshape(NFFT/2+1, 1, N_WIN, N_SAMP/(NFFT-noverlap)).swapaxes(0, 2)[:, :, :NFFT/2]

    ptt = lambda x: 10. * np.log10(specgram(np.pad(x.reshape(-1), pad_width=NFFT/2-1, mode='constant'),
                                            NFFT=NFFT,
                                            noverlap=noverlap)[0])
    X = np.array([ptt(x) for x in np.swapaxes(X, 0, 1)])
    X = np.reshape(X[:, :3*N_BINS], [N_FEA, N_FEA*N_BINS, N_WIN, N_BINS]).swapaxes(0, 2).reshape([N_WIN, 1, N_FEA*N_BINS, N_FEA*N_BINS])
    X = X - np.mean(X)
    return X

to_bool = lambda x: np.asarray(x, dtype=np.bool)

data_path = 'D:/PhD/Data/activity/Human Activity Recognition Using Smartphones Data Set V1/'
# files = glob(data_path + '/train/Inertial Signals/body_acc_*')
# Xtrain = pd.read_csv(files[0], sep=r'\s+')


def load_data():
    # LOAD DATA
    data = sio.loadmat('data/UCI_HAR_data.mat')

    X_train = spectro_conv(data['x_train'])
    y_train = pd.read_csv(data_path + '/train/y_train.txt', squeeze=True).values - 1
#    y_train = data['y_train']

    X_valid = spectro_conv(data['x_test'])
    y_valid = pd.read_csv(data_path + '/test/y_test.txt', squeeze=True).values - 1

    X_test = X_valid
    y_test = y_valid

    return dict(
        output_dim=int(np.unique(y_test).shape[0]),
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_height=X_train.shape[2],
        input_width=X_train.shape[3],
        )


def build_model(input_width, input_height, output_dim,
                batch_size=BATCH_SIZE):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, input_width, input_height),
        )

    # l_in_dropout = lasagne.layers.DropoutLayer(l_in, p=INPUTDROPOUT)

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=64,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    l_conv1_2 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters=64,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_2, pool_size=(2, 2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1,
        num_filters=128,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    l_conv2_2 = lasagne.layers.Conv2DLayer(
        l_conv2,
        num_filters=128,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_2, pool_size=(2, 2))

    # l_conv3 = lasagne.layers.Conv2DLayer(
    #     l_pool2,
    #     num_filters=256,
    #     filter_size=(3, 3),
    #     nonlinearity=lasagne.nonlinearities.rectify,
    #     W=lasagne.init.GlorotUniform(),
    #     )
    #
    # l_conv3_2 = lasagne.layers.Conv2DLayer(
    #     l_conv3,
    #     num_filters=256,
    #     filter_size=(3, 3),
    #     nonlinearity=lasagne.nonlinearities.rectify,
    #     W=lasagne.init.GlorotUniform(),
    #     )
    # l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3_2, pool_size=(2, 2))

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=DROPOUT)

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=256,
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

def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    pred = T.argmax(
        lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
        axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

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

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    return output_layer


if __name__ == '__main__':
    main()
