import theano
import theano.tensor as T
from lasagne import init
from models.base import Model
from lasagne_extensions.layers import (SampleLayer, MultinomialLogDensityLayer,
                                       GaussianLogDensityLayer, StandardNormalLogDensityLayer, BernoulliLogDensityLayer,
                                       InputLayer, DenseLayer, DimshuffleLayer, InverseLayer, ReshapeLayer,
                                       NonlinearityLayer, BatchNormLayer, get_all_params, get_output)
from lasagne_extensions.layers import (ConcatLayer, RepeatLayer, Conv2DLayer, MaxPool2DLayer, GlobalPoolLayer)
from lasagne_extensions.objectives import categorical_crossentropy, categorical_accuracy, aggregate, squared_error
from lasagne_extensions.nonlinearities import rectify, softplus, sigmoid, softmax
from lasagne_extensions.updates import total_norm_constraint
from lasagne_extensions.updates import rmsprop, adam, adagrad
from parmesan.distributions import log_normal
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne_extensions.layers.batch_norm import batch_norm as bn


class CVAE(Model):
    """
    Implementation of recurrent variational auto-encoder.
    """

    def __init__(self, n_x, n_z, qz_hid, px_hid, filters, seq_length=50, nonlinearity=rectify,
                 px_nonlinearity=None, x_dist='linear', batchnorm=False, seed=1234):
        """
        Weights are initialized using the Bengio and Glorot (2010) initialization scheme.
        :param n_x: Number of inputs.
        :param n_z: Number of latent.
        :param qz_hid: List of number of deterministic hidden q(z|a,x,y).
        :param px_hid: List of number of deterministic hidden p(a|z,y) & p(x|z,y).
        :param nonlinearity: The transfer function used in the deterministic layers.
        :param x_dist: The x distribution, 'bernoulli', 'multinomial', or 'gaussian'.
        :param batchnorm: Boolean value for batch normalization.
        :param seed: The random seed.
        """
        super(CVAE, self).__init__(n_x, qz_hid + px_hid, n_z, nonlinearity)
        self.x_dist = x_dist
        self.n_x = n_x
        self.seq_length = seq_length
        self.n_z = n_z
        self.batchnorm = batchnorm
        self._srng = RandomStreams(seed)

        # Pool layer cache
        pool_layers = []

        # Decide Glorot initializaiton of weights.
        init_w = 1e-3
        hid_w = ""
        if nonlinearity == rectify or nonlinearity == softplus:
            hid_w = "relu"

        # Define symbolic variables for theano functions.
        self.sym_x = T.tensor3('x')  # inputs
        self.sym_z = T.matrix('z')
        self.sym_samples = T.iscalar('samples')  # MC samples

        # Assist methods for collecting the layers
        def dense_layer(layer_in, n, dist_w=init.GlorotNormal, dist_b=init.Normal):
            dense = DenseLayer(layer_in, n, dist_w(hid_w), dist_b(init_w), None)
            if batchnorm:
                dense = bn(dense)
            return NonlinearityLayer(dense, self.transf)

        def stochastic_layer(layer_in, n, samples, nonlin=None):
            mu = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            logvar = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            return SampleLayer(mu, logvar, eq_samples=samples, iw_samples=1), mu, logvar

        def conv_layer(layer_in, filter, stride=(1, 1), pool=1, name='conv'):
            l_conv = Conv2DLayer(layer_in, num_filters=filter, filter_size=(3, 1), stride=stride, pad='full', name=name)
            if pool > 1:
                l_conv = MaxPool2DLayer(l_conv, pool_size=(pool, 1))
                pool_layers.append(l_conv)
            return l_conv

        # Reshape input
        l_x_in = InputLayer((None, seq_length, n_x), name='Input')
        l_x_in_reshp = ReshapeLayer(l_x_in, (-1, 1, seq_length, n_x))
        print("l_x_in_reshp", l_x_in_reshp.output_shape)

        # CNN encoder implementation
        l_conv_enc = l_x_in_reshp
        for filter, stride, pool in filters:
            l_conv_enc = conv_layer(l_conv_enc, filter, stride, pool)
            print("l_conv_enc", l_conv_enc.output_shape)

        # Pool along last 2 axes
        l_global_pool_enc = GlobalPoolLayer(l_conv_enc)
        l_enc = dense_layer(l_global_pool_enc, n_z)
        print("l_enc", l_enc.output_shape)

        # Recognition q(z|x)
        l_qz = l_enc
        for hid in qz_hid:
            l_qz = dense_layer(l_qz, hid)
        l_qz, l_qz_mu, l_qz_logvar = stochastic_layer(l_qz, n_z, self.sym_samples)
        print("l_qz", l_qz.output_shape)

        # Inverse pooling
        l_global_depool = InverseLayer(l_qz, l_global_pool_enc)
        print("l_global_depool", l_global_depool.output_shape)

        # Reverse pool layer order
        pool_layers = pool_layers[::-1]

        # Decode
        l_deconv = l_global_depool
        for idx, filter in enumerate(filters[::-1]):
            filter, stride, pool = filter
            if pool > 1:
                l_deconv = InverseLayer(l_deconv, pool_layers[idx])
            l_deconv = Conv2DLayer(l_deconv, num_filters=filter, filter_size=(3, 1), stride=(stride, 1), W=init.GlorotNormal('relu'))
            print("l_deconv", l_deconv.output_shape)

        # The last l_conv layer should give us the input shape
        l_dec = Conv2DLayer(l_deconv, num_filters=1, filter_size=(3, 1), pad='same', nonlinearity=None)
        print("l_dec", l_dec.output_shape)

        # Flatten first two dimensions
        l_dec = ReshapeLayer(l_dec, (-1, n_x))

        l_px = l_dec
        if x_dist == 'bernoulli':
            l_px = DenseLayer(l_px, n_x, init.GlorotNormal(), init.Normal(init_w), sigmoid)
        elif x_dist == 'multinomial':
            l_px = DenseLayer(l_px, n_x, init.GlorotNormal(), init.Normal(init_w), softmax)
        elif x_dist == 'gaussian':
            l_px, l_px_mu, l_px_logvar = stochastic_layer(l_px, n_x, self.sym_samples, px_nonlinearity)
        elif x_dist == 'linear':
            l_px = DenseLayer(l_px, n_x, nonlinearity=None)

        # Reshape all the model layers to have the same size
        self.l_x_in = l_x_in

        self.l_qz = ReshapeLayer(l_qz, (-1, self.sym_samples, 1, n_z))
        self.l_qz_mu = DimshuffleLayer(l_qz_mu, (0, 'x', 'x', 1))
        self.l_qz_logvar = DimshuffleLayer(l_qz_logvar, (0, 'x', 'x', 1))

        self.l_px = DimshuffleLayer(ReshapeLayer(l_px, (-1, seq_length, self.sym_samples, 1, n_x)), (0, 2, 3, 1, 4))
        self.l_px_mu = DimshuffleLayer(ReshapeLayer(l_px_mu, (-1, seq_length, self.sym_samples, 1, n_x)), (0, 2, 3, 1, 4)) \
            if x_dist == "gaussian" else None
        self.l_px_logvar = DimshuffleLayer(ReshapeLayer(l_px_logvar, (-1, seq_length, self.sym_samples, 1, n_x)), (0, 2, 3, 1, 4)) \
            if x_dist == "gaussian" else None

        # Predefined functions
        inputs = {self.l_x_in: self.sym_x}
        outputs = get_output(l_qz, inputs, deterministic=True)
        self.f_qz = theano.function([self.sym_x, self.sym_samples], outputs)

        inputs = {self.l_x_in: self.sym_x}
        outputs = get_output(self.l_px, inputs, deterministic=True).mean(axis=(1, 2))
        self.f_px = theano.function([self.sym_x, self.sym_samples], outputs)

        outputs = get_output(self.l_px_mu, inputs, deterministic=True).mean(axis=(1, 2))
        self.f_mu = theano.function([self.sym_x, self.sym_samples], outputs)

        outputs = get_output(self.l_px_logvar, inputs, deterministic=True).mean(axis=(1, 2))
        self.f_var = theano.function([self.sym_x, self.sym_samples], outputs)

        # Define model parameters
        self.model_params = get_all_params([self.l_px])
        self.trainable_model_params = get_all_params([self.l_px], trainable=True)

    def build_model(self, train_set, test_set, validation_set=None):
        """
        :param train_set_unlabeled: Unlabeled train set containing variables x, t.
        :param train_set_labeled: Unlabeled train set containing variables x, t.
        :param test_set: Test set containing variables x, t.
        :param validation_set: Validation set containing variables x, t.
        :return: train, test, validation function and dicts of arguments.
        """
        super(CVAE, self).build_model(train_set, test_set, validation_set)

        n = self.sh_train_x.shape[0].astype(theano.config.floatX)  # no. of data points

        # Define the layers for the density estimation used in the lower bound.
        l_log_qz = GaussianLogDensityLayer(self.l_qz, self.l_qz_mu, self.l_qz_logvar)
        l_log_pz = StandardNormalLogDensityLayer(self.l_qz)

        l_x_in = ReshapeLayer(self.l_x_in, (-1, self.seq_length * self.n_x))
        if self.x_dist == 'bernoulli':
            l_px = ReshapeLayer(self.l_px, (-1, self.sym_samples, 1, self.seq_length * self.n_x))
            l_log_px = BernoulliLogDensityLayer(l_px, l_x_in)
        elif self.x_dist == 'multinomial':
            l_px = ReshapeLayer(self.l_px, (-1, self.sym_samples, 1, self.seq_length * self.n_x))
            l_log_px = MultinomialLogDensityLayer(l_px, l_x_in)
        elif self.x_dist == 'gaussian':
            l_px_mu = ReshapeLayer(self.l_px_mu, (-1, self.sym_samples, 1, self.seq_length * self.n_x))
            l_px_logvar = ReshapeLayer(self.l_px_logvar, (-1, self.sym_samples, 1, self.seq_length * self.n_x))
            l_log_px = GaussianLogDensityLayer(l_x_in, l_px_mu, l_px_logvar)
        elif self.x_dist == 'linear':
            l_log_px = self.l_px

        self.sym_warmup = T.fscalar('warmup')
        def lower_bound(log_pz, log_qz, log_px):
            return log_px + (log_pz - log_qz)*(1. - self.sym_warmup - 0.1)

        # Lower bound
        out_layers = [l_log_pz, l_log_qz, l_log_px]
        inputs = {self.l_x_in: self.sym_x}
        out = get_output(out_layers, inputs, batch_norm_update_averages=False, batch_norm_use_averages=False)
        log_pz, log_qz, log_px = out

        # If the decoder output is linear we need the reconstruction error
        if self.x_dist == 'linear':
            log_px = -aggregate(squared_error(log_px.mean(axis=(1, 2)), self.sym_x), mode='mean')

        lb = lower_bound(log_pz, log_qz, log_px)
        lb = lb.mean(axis=(1, 2))  # Mean over the sampling dimensions

        if self.batchnorm:
            # TODO: implement the BN layer correctly.
            inputs = {self.l_x_in: self.sym_x}
            get_output(out_layers, inputs, weighting=None, batch_norm_update_averages=True, batch_norm_use_averages=False)

        # Regularizing with weight priors p(theta|N(0,1)), collecting and clipping gradients
        weight_priors = 0.0
        for p in self.trainable_model_params:
            if 'W' not in str(p):
                continue
            weight_priors += log_normal(p, 0, 1).sum()

        # Collect the lower bound and scale it with the weight priors.
        elbo = lb.mean()
        cost = (elbo * n + weight_priors) / -n

        grads_collect = T.grad(cost, self.trainable_model_params)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        clip_grad, max_norm = 1, 5
        mgrads = total_norm_constraint(grads_collect, max_norm=max_norm)
        mgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
        updates = adam(mgrads, self.trainable_model_params, self.sym_lr, sym_beta1, sym_beta2)
        # updates = rmsprop(mgrads, self.trainable_model_params, self.sym_lr + (0*sym_beta1*sym_beta2))

        # Training function
        x_batch = self.sh_train_x[self.batch_slice]
        if self.x_dist == 'bernoulli':  # Sample bernoulli input.
            x_batch = self._srng.binomial(size=x_batch.shape, n=1, p=x_batch, dtype=theano.config.floatX)

        givens = {self.sym_x: x_batch}
        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2, self.sym_samples, self.sym_warmup]
        outputs = [log_px.mean(), log_pz.mean(), log_qz.mean(), elbo]
        f_train = theano.function(inputs=inputs, outputs=outputs, givens=givens, updates=updates)

        # Default training args. Note that these can be changed during or prior to training.
        self.train_args['inputs']['batchsize'] = 100
        self.train_args['inputs']['learningrate'] = 1e-4
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        self.train_args['inputs']['samples'] = 1
        self.train_args['inputs']['warmup'] = 0
        self.train_args['outputs']['log p(x)'] = '%0.6f'
        self.train_args['outputs']['log p(z)'] = '%0.6f'
        self.train_args['outputs']['log q(z)'] = '%0.6f'
        self.train_args['outputs']['elbo train'] = '%0.6f'
        self.train_args['outputs']['warmup'] = '%0.3f'

        # Validation and test function
        givens = {self.sym_x: self.sh_test_x}
        f_test = theano.function(inputs=[self.sym_samples, self.sym_warmup], outputs=[elbo], givens=givens)

        # Test args.  Note that these can be changed during or prior to training.
        self.test_args['inputs']['samples'] = 1
        self.test_args['outputs']['elbo test'] = '%0.6f'

        f_validate = None
        if validation_set is not None:
            givens = {self.sym_x: self.sh_valid_x}
            f_validate = theano.function(inputs=[self.sym_samples], outputs=[elbo], givens=givens)
            # Default validation args. Note that these can be changed during or prior to training.
            self.validate_args['inputs']['samples'] = 1
            self.validate_args['outputs']['elbo validation'] = '%0.6f'

        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def get_output(self, x, samples=1):
        return self.f_px(x, samples)

    def model_info(self):
        s = ""
        s += 'batch norm: %s.\n' % (str(self.batchnorm))
        s += 'x distribution: %s.' % (str(self.x_dist))
        return s
