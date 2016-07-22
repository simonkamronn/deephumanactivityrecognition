import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from .base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne.layers import *
from lasagne import init
from lasagne_extensions.updates import adam, rmsprop
from lasagne_extensions.layers import TiedDropoutLayer
import numpy as np

CONST_FORGET_B = 1.
GRAD_CLIP = 5


class HRNN(Model):
    def __init__(self, n_in, n_hidden, n_out, l1_hidden,
                 grad_clip=GRAD_CLIP, peepholes=False, trans_func=rectify, out_func=softmax, factor=1,
                 output_dropout=0.0, slicers=()):
        super(HRNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length, n_features))
        l_prev = self.l_in
        print("Input shape", get_output_shape(l_prev))

        # Reshape
        sequence_length /= factor
        _shape = (-1, int(sequence_length), int(n_features))
        l_prev = ReshapeLayer(l_prev, shape=_shape)

        # Add input noise
        # self.log += "\nAdding noise layer: 0.05"
        # l_prev = GaussianNoiseLayer(l_prev, sigma=0.05)

        print("LSTM level 1 input shape", get_output_shape(l_prev))
        for i, n_hid in enumerate(l1_hidden):
            self.log += "\nAdding LSTM layer with: %d units" % n_hid
            l_prev = LSTMLayer(
                l_prev,
                num_units=n_hid,
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.GlorotNormal(),
                    W_hid=lasagne.init.Orthogonal()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=lasagne.nonlinearities.tanh,
                only_return_final=True if i==(len(l1_hidden)-1) else False)
        print("LSTM level 1 out shape", get_output_shape(l_prev))

        # Get output shapes
        s1, s2 = l_prev.output_shape

        # Reshape for second level LSTM
        l_prev = ReshapeLayer(l_prev, (-1, factor, s2))

        # Add LSTM layers
        print("LSTM level 2 input shape", get_output_shape(l_prev))
        for n_hid in n_hidden:
            self.log += "\nAdding LSTM layer with: %d units" % n_hid
            l_prev = LSTMLayer(
                l_prev,
                num_units=n_hid,
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.GlorotNormal(),
                    W_hid=lasagne.init.Orthogonal()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=lasagne.nonlinearities.tanh)
        print("LSTM level 2 output shape", l_prev.output_shape)

        # Reshape to process each timestep individually
        l_prev = ReshapeLayer(l_prev, (-1, n_hidden[-1]))
        # l_prev = DenseLayer(l_prev, num_units=512, nonlinearity=trans_func)
        # self.log += "\nAdding dense layer with %d units" % 512

        if output_dropout:
            self.log += "\nAdding output dropout with probability: %.2f" % output_dropout
            l_prev = DropoutLayer(l_prev, p=output_dropout)

        # Output
        l_prev = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model = ReshapeLayer(l_prev, (-1, factor, n_out))
        print("Output shape", get_output_shape(self.model))

        self.model_params = get_all_params(self.model)
        self.sym_x = T.tensor3('x')
        self.sym_t = T.tensor3('t')

    def build_model(self, train_set, test_set, validation_set=None, weights=None):
        super(HRNN, self).build_model(train_set, test_set, validation_set)

        def brier_score(given, predicted, weight_vector):
            return T.power(given - predicted, 2.0).dot(weight_vector).mean()

        epsilon = 1e-8
        y_train = T.clip(get_output(self.model, self.sym_x), epsilon, 1)
        loss_brier_train = brier_score(y_train, self.sym_t, weights)
        loss_cc = aggregate(categorical_crossentropy(y_train, self.sym_t), mode='mean')
        loss_train_acc = categorical_accuracy(y_train, self.sym_t).mean()

        y_test = T.clip(get_output(self.model, self.sym_x, deterministic=True), epsilon, 1)
        loss_brier_test = brier_score(y_test, self.sym_t, weights)
        loss_eval = aggregate(categorical_crossentropy(y_test, self.sym_t), mode='mean')
        loss_acc = categorical_accuracy(y_test, self.sym_t).mean()

        all_params = get_all_params(self.model, trainable=True)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        grads = T.grad(loss_brier_train, all_params)
        grads = [T.clip(g, -5, 5) for g in grads]
        updates = rmsprop(grads, all_params, self.sym_lr, sym_beta1, sym_beta2)

        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        f_train = theano.function(
            inputs, [loss_brier_train],
            updates=updates,
            givens={
                self.sym_x: self.sh_train_x[self.batch_slice],
                self.sym_t: self.sh_train_t[self.batch_slice],
            },
        )

        f_test = theano.function(
            [], [loss_brier_test],
            givens={
                self.sym_x: self.sh_test_x,
                self.sym_t: self.sh_test_t,
            },
            on_unused_input='ignore',
        )

        f_validate = None
        if validation_set is not None:
            f_validate = theano.function(
                [self.sym_batchsize], [loss_brier_test],
                givens={
                    self.sym_x: self.sh_valid_x,
                    self.sym_t: self.sh_valid_t,
                },
                on_unused_input='ignore',
            )

        predict = theano.function([self.sym_x], [y_test])

        self.train_args['inputs']['batchsize'] = 128
        self.train_args['inputs']['learningrate'] = 1e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        self.train_args['outputs']['loss_brier_train'] = '%0.6f'
        # self.train_args['outputs']['loss_train_acc'] = '%0.6f'

        # self.test_args['inputs']['batchsize'] = 128
        self.test_args['outputs']['loss_brier_test'] = '%0.6f'
        # self.test_args['outputs']['loss_acc'] = '%0.6f'

        # self.validate_args['inputs']['batchsize'] = 128
        # self.validate_args['outputs']['loss_eval'] = '%0.6f'
        # self.validate_args['outputs']['loss_acc'] = '%0.6f'
        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args, predict

    def model_info(self):
        return self.log

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)
