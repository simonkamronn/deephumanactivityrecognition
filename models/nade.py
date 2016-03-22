__author__ = 'larsma'

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from base import Model
from lasagne_extensions.objectives import binary_crossentropy
from lasagne_extensions.nonlinearities import sigmoid
from lasagne_extensions.updates import sgd
from lasagne_extensions.layers import NADELayer, InputLayer, get_all_params, get_output
from lasagne import init
import numpy as np

class NADE(Model):
    def __init__(self, n_v, n_h, trans_func=sigmoid):
        super(NADE, self).__init__(n_v, n_h, n_v, trans_func)
        self._srng = RandomStreams()
        self.n_hidden = n_h
        l_v = InputLayer((None, n_v))
        self.model = NADELayer(l_v, n_h, W=init.GlorotNormal(), b=init.Constant(0.))
        self.model_params = get_all_params(self.model)

        self.sym_x = T.matrix('x')

    def build_model(self, train_set, test_set, validation_set=None):
        super(NADE, self).build_model(train_set, test_set, validation_set)

        xhat = get_output(self.model, self.sym_x)
        loss = -((-binary_crossentropy(xhat, self.sym_x)).sum(axis=1)).mean()
        updates = sgd(loss, get_all_params(self.model), self.sym_lr)

        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr]
        x_batch = self.sh_train_x[self.batch_slice]
        x_batch = self._srng.binomial(size=x_batch.shape, n=1, p=x_batch, dtype=theano.config.floatX)
        givens = {self.sym_x: x_batch}
        f_train = theano.function(inputs, [loss], updates=updates, givens=givens)

        subset = 1000 # Only take a subset, in order not to receive memory errors.
        givens = {self.sym_x: self.sh_test_x[:subset]}
        f_test = theano.function([], [loss], givens=givens)

        f_validate = None
        if validation_set is not None:
            givens = {self.sym_x: self.sh_valid_x[:subset]}
            f_validate = theano.function([], [loss], givens=givens)

        self.train_args['inputs']['batchsize'] = 100
        self.train_args['inputs']['learningrate'] = 1e-2
        self.train_args['outputs']['like.'] = '%0.6f'
        self.test_args['outputs']['like.'] = '%0.6f'
        self.validate_args['outputs']['like.'] = '%0.6f'
        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def sample(self, num_samples):
        return self.model.sample(num_samples)

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)