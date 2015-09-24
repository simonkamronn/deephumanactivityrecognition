__author__ = 'Simon'

import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from base import Model
from deepmodels.nonlinearities import rectify, softmax
from lasagne.layers import get_output, LSTMLayer, Gate, ConcatLayer, DenseLayer, DropoutLayer, InputLayer, SliceLayer
import numpy as np
CONST_FORGET_B = 1.
GRAD_CLIP = 5

class RNN(Model):
    def __init__(self, n_in, n_hidden, n_out, downsample=1, ccf=False, grad_clip=GRAD_CLIP,
                 trans_func=rectify, out_func=softmax, batch_size=100, dropout_probability=0.0):
        super(RNN, self).__init__(n_in, n_hidden, n_out, batch_size, trans_func)
        self.outf = out_func
        self.n_layers = len(n_hidden)
        self.x = T.tensor3('x')
        self.t = T.matrix('t')

        # Define model using lasagne framework
        dropout = True if not dropout_probability == 0.0 else False

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(batch_size, sequence_length, n_features))
        l_prev = self.l_in

        # Data for getting output shape
        x = np.random.random((batch_size, sequence_length, n_features)).astype(theano.config.floatX)
        sym_x = T.tensor3('x')

        # Downsample input
        if downsample > 1:
            print("Downsampling with a factor of %d" % downsample)
            l_prev = lasagne.layers.FeaturePoolLayer(l_prev, pool_size=downsample, pool_function=T.mean)
            sequence_length /= downsample

        if ccf:
            print("Adding cross-channel feature layer")
            l_prev = lasagne.layers.ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features))
            l_prev = lasagne.layers.Conv2DLayer(l_prev,
                                                num_filters=n_features,
                                                filter_size=(1, n_features),
                                                nonlinearity=None)
            l_prev = lasagne.layers.ReshapeLayer(l_prev, (batch_size, n_features, sequence_length))
            l_prev = lasagne.layers.DimshuffleLayer(l_prev, (0, 2, 1))

        print("LSTM input shape", lasagne.layers.get_output(l_prev, sym_x).eval({sym_x: x}).shape)
        for n_hid in n_hidden:
            print("Adding BLSTM layer with %d units" % n_hid)
            l_forward = LSTMLayer(
                l_prev,
                num_units=n_hid/2,
                grad_clipping=grad_clip,
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=lasagne.nonlinearities.tanh
            )
            l_backward = LSTMLayer(
                l_prev,
                num_units=n_hid/2,
                grad_clipping=grad_clip,
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
        l_prev = ConcatLayer(
            [l_forward_slice, l_backward_slice],
            axis=1
        )

        if dropout:
            print("Adding output dropout with probability %.2f" % dropout_probability)
            l_prev = DropoutLayer(l_prev, p=dropout_probability)

        print("Output input shape", get_output(l_prev, sym_x).eval({sym_x: x}).shape)
        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)

    def build_model(self, *args):
        super(RNN, self).build_model(*args)

        epsilon = 1e-8
        loss_train = self.loss(
            T.clip(
                lasagne.layers.get_output(self.model, self.x),
                epsilon,
                1),
            self.t
        ).mean()

        loss_eval = self.loss(
            T.clip(
                lasagne.layers.get_output(self.model, self.x, deterministic=True),
                epsilon,
                1),
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
