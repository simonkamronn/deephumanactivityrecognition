import theano
import theano.tensor as T
from lasagne import init
from models.base import Model
from lasagne_extensions.layers import (InputLayer, DenseLayer, ReshapeLayer,
                                       NonlinearityLayer, get_all_params, get_output, get_all_param_values)
from lasagne_extensions.layers import (RecurrentLayer, LSTMLayer, ConcatLayer, RepeatLayer, Gate)
from lasagne_extensions.objectives import categorical_crossentropy, categorical_accuracy, squared_error, aggregate
from lasagne_extensions.nonlinearities import rectify, softplus, sigmoid, softmax
from lasagne_extensions.updates import total_norm_constraint
from lasagne_extensions.updates import rmsprop, adam, adagrad
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.regularization import regularize_network_params, l2
from lasagne.layers import BatchNormLayer
import pickle
from utils import env_paths as paths


class RAE(Model):
    """
    Implementation of variational recurrent auto-encoder.
    """

    def __init__(self, n_c, px_hid, enc_rnn=256, dec_rnn=256, n_l=50,
                 nonlinearity=rectify, batchnorm=False, seed=1234):
        """
        Weights are initialized using the Bengio and Glorot (2010) initialization scheme.
        :param n_c: Number of inputs.
        :param px_hid: List of number of deterministic hidden p(a|z,y) & p(x|z,y).
        :param nonlinearity: The transfer function used in the deterministic layers.
        :param x_dist: The x distribution, 'bernoulli', 'multinomial', or 'gaussian'.
        :param batchnorm: Boolean value for batch normalization.
        :param seed: The random seed.
        """
        super(RAE, self).__init__(n_c, px_hid, enc_rnn, nonlinearity)
        self.n_x = n_c
        self.max_seq_length = n_l
        self.batchnorm = batchnorm
        self._srng = RandomStreams(seed)

        # Decide Glorot initializaiton of weights.
        init_w = 1e-3
        hid_w = ""
        if nonlinearity == rectify or nonlinearity == softplus:
            hid_w = "relu"

        # Define symbolic variables for theano functions.
        self.sym_x = T.tensor3('x')  # inputs

        # Assist methods for collecting the layers
        def dense_layer(layer_in, n, dist_w=init.GlorotNormal, dist_b=init.Normal):
            dense = DenseLayer(layer_in, n, dist_w(hid_w), dist_b(init_w), nonlinearity=None)
            if batchnorm:
                dense = BatchNormLayer(dense)
            return NonlinearityLayer(dense, self.transf)

        def lstm_layer(input, nunits, return_final, backwards=False, name='LSTM'):
            ingate=Gate(W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01), b=init.Constant(0.0))
            forgetgate=Gate(W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01), b=init.Constant(5.0))
            cell=Gate(W_cell=None, nonlinearity=T.tanh, W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01),)
            outgate=Gate(W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01), b=init.Constant(0.0))

            lstm = LSTMLayer(input, num_units=nunits, backwards=backwards,
                             peepholes=False,
                             ingate=ingate,
                             forgetgate=forgetgate,
                             cell=cell,
                             outgate=outgate,
                             name=name,
                             only_return_final=return_final)

            rec = RecurrentLayer(input, num_units=nunits,
                                 W_in_to_hid=init.GlorotNormal('relu'),
                                 W_hid_to_hid=init.GlorotNormal('relu'), backwards=backwards,
                                 nonlinearity=rectify, only_return_final=return_final, name=name)
            return lstm

        # RNN encoder implementation
        l_x_in = InputLayer((None, None, n_c))
        l_enc_forward = lstm_layer(l_x_in, enc_rnn, return_final=True, backwards=False, name='enc_forward')
        l_enc_backward = lstm_layer(l_x_in, enc_rnn, return_final=True, backwards=True, name='enc_backward')
        l_enc_concat = ConcatLayer([l_enc_forward, l_enc_backward], axis=-1)
        l_enc = dense_layer(l_enc_concat, enc_rnn)

        # RNN decoder implementation
        l_dec_repeat = RepeatLayer(l_enc, n=n_l)
        l_dec_forward = lstm_layer(l_dec_repeat, dec_rnn, return_final=False, backwards=False, name='dec_forward')
        l_dec_backward = lstm_layer(l_dec_repeat, dec_rnn, return_final=False, backwards=True, name='dec_backward')
        l_dec_concat = ConcatLayer([l_dec_forward, l_dec_backward], axis=-1)
        l_dec = ReshapeLayer(l_dec_concat, (-1, 2*dec_rnn))
        l_dec = dense_layer(l_dec, dec_rnn)

        # Generative p(x_hat|x)
        l_px = l_dec
        for hid in px_hid:
            l_px = dense_layer(l_px, hid)

        # Output
        self.l_enc = l_enc

        l_px = DenseLayer(l_px, n_c, nonlinearity=None)
        self.l_px = ReshapeLayer(l_px, (-1, n_l, n_c))
        self.l_x_in = l_x_in

        inputs = {l_x_in: self.sym_x}
        outputs = get_output(self.l_px, inputs, deterministic=True)
        self.f_px = theano.function([self.sym_x], outputs, on_unused_input='warn')

        # Define model parameters
        self.encoder_params = get_all_param_values(self.l_enc)
        self.model_params = get_all_params(self.l_px)
        self.trainable_model_params = get_all_params(self.l_px, trainable=True)

    def build_model(self, train_set, test_set, validation_set=None):
        """
        :param train_set_unlabeled: Unlabeled train set containing variables x, t.
        :param train_set_labeled: Unlabeled train set containing variables x, t.
        :param test_set: Test set containing variables x, t.
        :param validation_set: Validation set containing variables x, t.
        :return: train, test, validation function and dicts of arguments.
        """
        super(RAE, self).build_model(train_set, test_set, validation_set)

        # Cost
        inputs = {self.l_x_in: self.sym_x}
        # px = get_output(self.l_px, inputs, batch_norm_update_averages=False, batch_norm_use_averages=False)
        px = get_output(self.l_px, inputs)
        cost = aggregate(squared_error(px, self.sym_x), mode='mean')
        # cost += 1e-4 * regularize_network_params(self.l_px, l2)

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

        givens = {self.sym_x: x_batch}
        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        outputs = [cost]
        f_train = theano.function(inputs=inputs, outputs=outputs, givens=givens, updates=updates)

        # Default training args. Note that these can be changed during or prior to training.
        self.train_args['inputs']['batchsize'] = 100
        self.train_args['inputs']['learningrate'] = 3e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        self.train_args['outputs']['cost train'] = '%0.6f'

        # Validation and test function
        givens = {self.sym_x: self.sh_test_x}
        f_test = theano.function(inputs=[], outputs=[cost], givens=givens)

        # Test args.  Note that these can be changed during or prior to training.
        self.test_args['outputs']['cost test'] = '%0.6f'

        f_validate = None
        if validation_set is not None:
            givens = {self.sym_x: self.sh_valid_x}
            f_validate = theano.function(inputs=[], outputs=[cost], givens=givens)

            # Default validation args. Note that these can be changed during or prior to training.
            self.validate_args['outputs']['cost val'] = '%0.6f'

        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def get_output(self, x):
        return self.f_px(x)

    def model_info(self):
        s = ""
        s += 'batch norm: %s.\n' % (str(self.batchnorm))
        return s

    def save_encoder(self):
        """
        Dump the model into a pickled version
        """
        p = paths.path_exists(self.root_path + '/pickled model/')
        p += 'encoder.pkl'
        pickle.dump(self.encoder_params, open(p, "wb"), protocol=pickle.HIGHEST_PROTOCOL)