__author__ = 'Simon'

import theano
import theano.tensor as T
import lasagne
from base import Model
from deepmodels.nonlinearities import rectify, softmax


class CNN(Model):
    def __init__(self, n_in, n_filters, filter_sizes, n_out, pool_size=2, n_hidden=512, downsample=1, ccf=False,
                 sum_channels=False, trans_func=rectify, out_func=softmax, batch_size=100, dropout_probability=0.0):
        super(CNN, self).__init__(n_in, n_hidden, n_out, batch_size, trans_func)
        self.outf = out_func
        self.n_layers = len(n_filters)
        self.x = T.tensor3('x')
        self.t = T.matrix('t')

        # Define model using lasagne framework
        dropout = True if not dropout_probability == 0.0 else False

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = lasagne.layers.InputLayer(shape=(batch_size, sequence_length, n_features))
        l_prev = self.l_in

        # Downsample input
        if downsample > 1:
            print("Downsampling with a factor of %d" % downsample)
            l_prev = lasagne.layers.FeaturePoolLayer(l_prev, pool_size=downsample, pool_function=T.mean)
            sequence_length /= downsample

        if ccf:
            print("Adding cross-channel feature layer")
            l_prev = lasagne.layers.ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features))
            l_prev = lasagne.layers.Conv2DLayer(l_prev, num_filters=n_features, filter_size=(1, n_features), nonlinearity=None)
            l_prev = lasagne.layers.ReshapeLayer(l_prev, (batch_size, n_features, sequence_length))
            l_prev = lasagne.layers.DimshuffleLayer(l_prev, (0, 2, 1))

        if sum_channels:
            l_prev = lasagne.layers.DimshuffleLayer(l_prev, (0, 2, 1))
            for n_filter, filter_size in zip(n_filters, filter_sizes):
                print("Adding conv layer: %d x %d" % (n_filter, filter_size))
                l_tmp = lasagne.layers.Conv1DLayer(l_prev, num_filters=n_filter, filter_size=filter_size, nonlinearity=self.transf)
                if pool_size > 1:
                    print("Adding max pooling layer: %d" % pool_size)
                    l_tmp = lasagne.layers.MaxPool1DLayer(l_tmp, pool_size=pool_size)
                l_prev = l_tmp
        else:
            l_prev = lasagne.layers.ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features))
            for n_filter, filter_size in zip(n_filters, filter_sizes):
                print("Adding 2D conv layer: %d x %d" % (n_filter, filter_size))
                l_tmp = lasagne.layers.Conv2DLayer(l_prev, num_filters=n_filter, filter_size=(filter_size, 1), nonlinearity=self.transf)
                if pool_size > 1:
                    print("Adding max pooling layer: %d" % pool_size)
                    l_tmp = lasagne.layers.MaxPool2DLayer(l_tmp, pool_size=(pool_size, 1))
                l_prev = l_tmp

        print("Adding dense layer with %d units" % n_hidden)
        l_prev = lasagne.layers.DenseLayer(l_prev, num_units=n_hidden, nonlinearity=self.transf)
        if dropout:
            l_prev = lasagne.layers.DropoutLayer(l_prev, p=dropout_probability)

        self.model = lasagne.layers.DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)

    def build_model(self, *args):
        super(CNN, self).build_model(*args)

        epsilon = 1e-8
        loss_train = self.loss(
            T.clip(
                lasagne.layers.get_output(self.model, self.x),
                epsilon,
                1-epsilon),
            self.t
        ).mean()

        loss_eval = self.loss(
            T.clip(
                lasagne.layers.get_output(self.model, self.x, deterministic=True),
                epsilon,
                1-epsilon),
            self.t
        ).mean()

        accuracy = T.mean(T.eq(
            T.argmax(lasagne.layers.get_output(self.model, self.x, deterministic=True), axis=1),
            T.argmax(self.t, axis=1)
        ), dtype=theano.config.floatX)

        all_params = lasagne.layers.get_all_params(self.model)
        updates = self.update(loss_train, all_params, *self.update_args)

        train_model = theano.function(
            [self.index], loss_train,
            updates=updates,
            givens={
                self.x: self.train_x[self.batch_slice],
                self.t: self.train_t[self.batch_slice],
            },
        )

        test_model = theano.function(
            [self.index], loss_eval,
            givens={
                self.x: self.test_x[self.batch_slice],
                self.t: self.test_t[self.batch_slice],
            },
        )

        validate_model = None
        if self.validation_x is not None:
            validate_model = theano.function(
                [self.index], [loss_eval, accuracy],
                givens={
                    self.x: self.validation_x[self.batch_slice],
                    self.t: self.validation_t[self.batch_slice],
                },
            )
        return train_model, test_model, validate_model
