import theano
theano.config.floatX = 'float32'
import theano.tensor as T
from .base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, get_output_shape, DenseLayer, DropoutLayer, InputLayer, SliceLayer, \
    ReshapeLayer, DimshuffleLayer, get_all_params, Conv2DLayer, Pool2DLayer, GlobalPoolLayer, ConcatLayer, \
    GaussianNoiseLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne import init
from lasagne.layers import BatchNormLayer
from lasagne_extensions.updates import rmsprop
from lasagne_extensions.layers import TiedDropoutLayer


class CNN(Model):
    def __init__(self, n_in, n_filters, filter_sizes, n_out, pool_sizes=None, n_hidden=(512),
                 trans_func=rectify, out_func=softmax, dropout=0.0, input_noise=0.0,
                 batch_norm=False, conv_dropout=0.0, slicers=()):
        super(CNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        if pool_sizes is None:
            pool_sizes = [0]*len(n_filters)

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length, n_features))
        l_prev = self.l_in

        # Apply input noise
        # l_prev = GaussianNoiseLayer(l_prev, sigma=input_noise)

        # 2D Convolutional layers
        l_prev = ReshapeLayer(l_prev, (-1, 1, sequence_length, n_features))
        l_prev = DimshuffleLayer(l_prev, (0, 3, 2, 1))

        # Add the convolutional filters
        for n_filter, filter_size, pool_size in zip(n_filters, filter_sizes, pool_sizes):
            self.log += "\nAdding 2D conv layer: %d x %d" % (n_filter, filter_size)
            l_prev = Conv2DLayer(l_prev,
                                 num_filters=n_filter,
                                 filter_size=(filter_size, 1),
                                 nonlinearity=self.transf,
                                 pad=filter_size//2)
            if batch_norm:
                l_prev = BatchNormLayer(l_prev)
            if pool_size > 1:
                self.log += "\nAdding max pooling layer: %d" % pool_size
                l_prev = Pool2DLayer(l_prev, pool_size=(pool_size, 1))
            self.log += "\nAdding dropout layer: %.2f" % conv_dropout
            l_prev = TiedDropoutLayer(l_prev, p=conv_dropout)
            print("Conv out shape", get_output_shape(l_prev))

        # Global pooling layer
        l_prev = GlobalPoolLayer(l_prev, pool_function=T.mean, name='Global Mean Pool')
        print("GlobalPoolLayer out shape", get_output_shape(l_prev))

        for n_hid in n_hidden:
            self.log += "\nAdding dense layer with %d units" % n_hid
            print("Dense input shape", get_output_shape(l_prev))
            l_prev = DenseLayer(l_prev, n_hid, init.GlorotNormal(), init.Normal(1e-3), self.transf)
            if batch_norm:
                l_prev = BatchNormLayer(l_prev)
            if dropout:
                self.log += "\nAdding dense dropout with probability: %.2f" % dropout
                l_prev = DropoutLayer(l_prev, p=dropout)

        if batch_norm:
            self.log += "\nUsing batch normalization"

        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None, weights=None):
        super(CNN, self).build_model(train_set, test_set, validation_set)

        def brier_score(given, predicted, weight_vector):
            return T.power(given - predicted, 2.0).dot(weight_vector).mean()

        epsilon = 1e-8
        y_train = T.clip(get_output(self.model, self.sym_x), epsilon, 1)
        train_brier = brier_score(y_train, self.sym_t, weights)
        train_cc = aggregate(categorical_crossentropy(y_train, self.sym_t), mode='mean')
        loss_train_acc = categorical_accuracy(y_train, self.sym_t).mean()

        y_test = T.clip(get_output(self.model, self.sym_x, deterministic=True), epsilon, 1)
        test_brier = brier_score(y_test, self.sym_t, weights)
        test_cc = aggregate(categorical_crossentropy(y_test, self.sym_t), mode='mean')
        test_acc = categorical_accuracy(y_test, self.sym_t).mean()

        all_params = get_all_params(self.model, trainable=True)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        grads = T.grad(train_cc, all_params)
        grads = [T.clip(g, -5, 5) for g in grads]
        updates = rmsprop(grads, all_params, self.sym_lr, sym_beta1, sym_beta2)

        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        f_train = theano.function(
            inputs, [train_cc, train_brier],
            updates=updates,
            givens={
                self.sym_x: self.sh_train_x[self.batch_slice],
                self.sym_t: self.sh_train_t[self.batch_slice],
            },
        )

        f_test = theano.function(
            [], [test_cc, test_brier],
            givens={
                self.sym_x: self.sh_test_x,
                self.sym_t: self.sh_test_t,
            },
        )

        f_validate = None
        if validation_set is not None:
            f_validate = theano.function(
                [self.sym_index, self.sym_batchsize], [test_cc, test_acc],
                givens={
                    self.sym_x: self.sh_valid_x[self.batch_slice],
                    self.sym_t: self.sh_valid_t[self.batch_slice],
                },
            )

        predict = theano.function([self.sym_x], [y_test])

        self.train_args['inputs']['batchsize'] = 64
        self.train_args['inputs']['learningrate'] = 1e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 1e-6
        self.train_args['outputs']['train_cc'] = '%0.4f'
        # self.train_args['outputs']['train_acc'] = '%0.4f'
        self.train_args['outputs']['train_brier'] = '%0.4f'

        # self.test_args['inputs']['batchsize'] = 64
        self.test_args['outputs']['test_cc'] = '%0.4f'
        # self.test_args['outputs']['test_acc'] = '%0.4f'
        self.test_args['outputs']['test_brier'] = '%0.4f'

        # self.validate_args['inputs']['batchsize'] = 64
        # self.validate_args['outputs']['loss_eval'] = '%0.6f'
        # self.validate_args['outputs']['test_acc'] = '%0.6f'
        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args, predict

    def model_info(self):
        return self.log

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)
