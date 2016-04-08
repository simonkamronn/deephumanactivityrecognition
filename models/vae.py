import theano
import theano.tensor as T
from lasagne import init
from .base import Model
from lasagne_extensions.layers import (StandardNormalLogDensityLayer, MultinomialLogDensityLayer,
                                       SampleLayer, GaussianLogDensityLayer, BernoulliLogDensityLayer,
                                       InputLayer, DenseLayer, DimshuffleLayer, ElemwiseSumLayer, ReshapeLayer,
                                       NonlinearityLayer, get_all_params, get_output, get_output_shape, get_all_layers)
from lasagne_extensions.nonlinearities import rectify, sigmoid, softmax
from lasagne_extensions.updates import total_norm_constraint
from lasagne_extensions.updates import adam
from lasagne_extensions import logdists
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np


class VAE(Model):
    """
    The :class:'VAE' class represents a unsupervised model following the article on variational autoencoders as a
    deep generative models. http://arxiv.org/abs/1312.6114.
    """

    def __init__(self, n_x, n_z, z_hidden, xhat_hidden, trans_func=rectify, init_w=1e-3,
                 x_dist='gaussian', batchnorm=False):
        super(VAE, self).__init__(n_x, z_hidden + xhat_hidden, n_z, trans_func)
        self.n_x = n_x
        self.n_z = n_z
        self.x_dist = x_dist
        self.batchnorm = batchnorm
        self.sym_x = T.matrix('x')  # symbolic inputs
        self.sym_z = T.matrix('z')
        self.sym_samples = T.iscalar('samples')
        self._srng = RandomStreams()

        def stochastic_layer(layer_in, n, samples, nonlin=None):
            mu = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            logvar = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            return SampleLayer(mu, logvar, eq_samples=samples, iw_samples=1), mu, logvar

        # Input
        l_x_in = InputLayer((None, n_x))

        # Inference q(z|x)
        l_z_x = l_x_in
        for hid in z_hidden:
            l_z_x = DenseLayer(l_z_x, hid, init.Normal(std=init_w), init.Normal(std=init_w), self.transf)
        l_z_x, l_z_x_mu, l_z_x_logvar = stochastic_layer(l_z_x, n_z, self.sym_samples)

        # Reshape for density layers
        l_z_x_reshaped = ReshapeLayer(l_z_x, (-1, self.sym_samples, n_z))
        l_z_x_mu_reshaped = DimshuffleLayer(l_z_x_mu, (0, 'x', 1))
        l_z_x_logvar_reshaped = DimshuffleLayer(l_z_x_logvar, (0, 'x', 1))

        # Generative p(xhat|z)
        l_xhat_z = l_z_x
        for hid in xhat_hidden:
            l_xhat_z = DenseLayer(l_xhat_z, hid, init.Normal(std=init_w), init.Normal(std=init_w), self.transf)
        if x_dist == 'bernoulli':
            l_xhat_z_mu_reshaped = None
            l_xhat_z_logvar_reshaped = None
            l_xhat_z = DenseLayer(l_xhat_z, n_x, init.Normal(std=init_w), init.Normal(std=init_w), sigmoid)
        elif x_dist == 'gaussian':
            l_xhat_z, l_xhat_z_mu, l_xhat_z_logvar = stochastic_layer(l_xhat_z, n_x, self.sym_samples)
            l_xhat_z_mu_reshaped = ReshapeLayer(l_xhat_z_mu, (-1, self.sym_samples, 1, n_x))
            l_xhat_z_logvar_reshaped = ReshapeLayer(l_xhat_z_logvar, (-1, self.sym_samples, 1, n_x))
        l_xhat_z_reshaped = ReshapeLayer(l_xhat_z, (-1, self.sym_samples, 1, n_x))

        # Init class variables
        self.l_x_in = l_x_in
        self.l_xhat_mu = l_xhat_z_mu_reshaped
        self.l_xhat_logvar = l_xhat_z_logvar_reshaped
        self.l_xhat = l_xhat_z_reshaped
        self.l_z = l_z_x_reshaped
        self.l_z_mu = l_z_x_mu_reshaped
        self.l_z_logvar = l_z_x_logvar_reshaped
        self.model_params = get_all_params(self.l_xhat)

        inputs = [self.sym_x, self.sym_samples]
        outputs = get_output(self.l_z, self.sym_x, deterministic=True).mean(axis=1)
        self.f_qz = theano.function(inputs, outputs)

        inputs = {l_z_x: self.sym_z}
        outputs = get_output(self.l_xhat, inputs, deterministic=True).mean(axis=(1, 2))
        inputs = [self.sym_z, self.sym_samples]
        self.f_px = theano.function(inputs, outputs)

    def build_model(self, train_set, test_set, validation_set=None):
        super(VAE, self).build_model(train_set, test_set, validation_set)

        # Density estimations
        l_log_pz = StandardNormalLogDensityLayer(self.l_z)
        l_log_qz_x = GaussianLogDensityLayer(self.l_z, self.l_z_mu, self.l_z_logvar)
        if self.x_dist == 'bernoulli':
            l_px_z = BernoulliLogDensityLayer(self.l_xhat, self.l_x_in)
        elif self.x_dist == 'gaussian':
            l_px_z = GaussianLogDensityLayer(self.l_x_in, self.l_xhat_mu, self.l_xhat_logvar)

        out_layers = [l_log_pz, l_log_qz_x, l_px_z]
        inputs = {self.l_x_in: self.sym_x}
        log_pz, log_qz_x, log_px_z = get_output(out_layers, inputs)
        lb = -(log_pz + log_px_z - log_qz_x).mean(axis=1).mean()

        all_params = get_all_params(self.l_xhat, trainable=True)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        updates = adam(lb, all_params, self.sym_lr, sym_beta1, sym_beta2)

        x_batch = self.sh_train_x[self.batch_slice]
        if self.x_dist == 'bernoulli':
            x_batch = self._srng.binomial(size=x_batch.shape, n=1, p=x_batch, dtype=theano.config.floatX)
        givens = {self.sym_x: x_batch}
        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2, self.sym_samples]
        outputs = [lb]
        f_train = theano.function(inputs=inputs, outputs=outputs, givens=givens, updates=updates)
        # Training args
        self.train_args['inputs']['batchsize'] = 100
        self.train_args['inputs']['learningrate'] = 3e-4
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        self.train_args['inputs']['samples'] = 1
        self.train_args['outputs']['lb'] = '%0.4f'

        givens = {self.sym_x: self.sh_test_x}
        inputs = [self.sym_samples]
        outputs = [lb]
        f_test = theano.function(inputs=inputs, outputs=outputs, givens=givens)
        # Testing args
        self.test_args['inputs']['samples'] = 1
        self.test_args['outputs']['lb'] = '%0.4f'

        f_validate = None
        if validation_set is not None:
            givens = {self.sym_x: self.sh_valid_x}
            inputs = [self.sym_samples]
            outputs = [lb]
            f_validate = theano.function(inputs=inputs, outputs=outputs, givens=givens)
            # Validation args
            self.validate_args['inputs']['samples'] = 1
            self.validate_args['outputs']['lb'] = '%0.4f'

        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def draw_sample(self, z=None, samples=1, n=100):
        """
        Draw a sample from the latent z distribution.
        :param z: if given, this will forward propagate through the generative model.
        :return: the reconstructed x.
        """
        if z is None:  # draw random z
            z = np.asarray(np.random.normal(0, 1, size=(n, self.n_z)), dtype=theano.config.floatX)
        return self.f_xhat(z, samples)

    def get_output(self, x, samples=1):
        return self.f_z(x, samples)
