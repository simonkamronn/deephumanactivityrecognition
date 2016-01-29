import theano
theano.config.floatX = 'float32'
import theano.tensor as T
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, get_output_shape, DenseLayer, InputLayer, ConcatLayer, \
    ReshapeLayer, DimshuffleLayer, get_all_params, Conv2DLayer, Pool2DLayer, GlobalPoolLayer, SliceLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
# from lasagne.layers.dnn import Conv2DDNNLayer as
# from lasagne.layers.dnn import Pool2DDNNLayer as
from lasagne_extensions.updates import adam, rmsprop, nesterov_momentum
from modules import ResidualModule, BatchNormalizeLayer
from lasagne_extensions.layers import TiedDropoutLayer


class ResNet(Model):
    def __init__(self, n_in, n_filters, n_out, pool_sizes=None, n_hidden=(), ccf=False,
                 conv_dropout=0.0, trans_func=rectify, out_func=softmax, dropout=0.0,
                 batch_norm=False, stats=2):
        super(ResNet, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length+stats, n_features), name='Input')
        l_prev = self.l_in
        print("Input shape: ", get_output_shape(l_prev))

        # Separate into raw values and statistics
        stats_layer = SliceLayer(l_prev, indices=slice(sequence_length, None), axis=1)
        stats_layer = ReshapeLayer(stats_layer, (-1, stats*n_features))
        l_prev = SliceLayer(l_prev, indices=slice(0, sequence_length), axis=1)

        if ccf:
            self.log += "\nAdding cross-channel feature layer"
            l_prev = ReshapeLayer(l_prev, (-1, 1, sequence_length, n_features), name='Reshape')
            l_prev = Conv2DLayer(l_prev,
                                 num_filters=4*n_features,
                                 filter_size=(1, n_features),
                                 nonlinearity=None,
                                 b=None,
                                 name='Conv2D')
            l_prev = BatchNormalizeLayer(l_prev, normalize=batch_norm, nonlinearity=self.transf)
            n_features *= 4
            l_prev = ReshapeLayer(l_prev, (-1, n_features, sequence_length), name='Reshape')
            l_prev = DimshuffleLayer(l_prev, (0, 2, 1), name='Dimshuffle')

        # Reshape and order dimensionality
        l_prev = ReshapeLayer(l_prev, (-1, 1, sequence_length, n_features), name='Reshape')
        l_prev = DimshuffleLayer(l_prev, (0, 3, 2, 1), name='Dimshuffle')

        # Convolutional layers
        n_filter, filter_size, pool_size = (16, 3, 2)
        self.log += "\nAdding 2D conv layer: %d x %d" % (n_filter, filter_size)
        l_prev = Conv2DLayer(l_prev,
                             num_filters=n_filter,
                             filter_size=(filter_size, 1),
                             stride=(1, 1),
                             pad="same",
                             nonlinearity=None,
                             b=None,
                             name='Conv2D')
        l_prev = BatchNormalizeLayer(l_prev, normalize=batch_norm, nonlinearity=self.transf)
        if pool_size > 1:
            self.log += "\nAdding max pooling layer: %d" % pool_size
            l_prev = Pool2DLayer(l_prev, pool_size=(pool_size, 1), name='Max Pool')
        if conv_dropout > 0.0:
            self.log += "\nAdding dropout layer: %.2f" % conv_dropout
            l_prev = TiedDropoutLayer(l_prev, p=conv_dropout, name='Tied Dropout')
        print("Conv out shape", get_output_shape(l_prev))

        # Residual modules
        for n_filter, pool_size in zip(n_filters, pool_sizes):
            self.log += "\nAdding residual module: %d" % n_filter
            l_prev = ResidualModule(l_prev,
                                    num_filters=n_filter,
                                    nonlinearity=self.transf,
                                    normalize=batch_norm,
                                    stride=(pool_size, 1),
                                    conv_dropout=conv_dropout)
            # if pool_size > 2:
            #     self.log += "\nAdding max pool layer: %d" % pool_size
            #     l_prev = Pool2DLayer(l_prev, pool_size=(pool_size, 1), name='Max Pool')
            print("Residual out shape", get_output_shape(l_prev))

        self.log += "\nGlobal Pooling: mean"
        l_prev = GlobalPoolLayer(l_prev, pool_function=T.mean, name='Global pooling layer: mean')
        print("GlobalPoolLayer out shape", get_output_shape(l_prev))

        # Append statistics
        l_prev = ConcatLayer((l_prev, stats_layer), axis=1)

        for n_hid in n_hidden:
            self.log += "\nAdding dense layer with %d units" % n_hid
            print("Dense input shape", get_output_shape(l_prev))
            l_prev = DenseLayer(l_prev, num_units=n_hid, nonlinearity=None, b=None, name='Dense')
            l_prev = BatchNormalizeLayer(l_prev, normalize=batch_norm, nonlinearity=self.transf)
            if dropout > 0.0:
                self.log += "\nAdding output dropout with probability %.2f" % dropout
                l_prev = TiedDropoutLayer(l_prev, p=dropout, name='Dropout')

        if batch_norm:
            self.log += "\nUsing batch normalization"

        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func, name='Output')
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(ResNet, self).build_model(train_set, test_set, validation_set)

        epsilon = 1e-8
        y_train = T.clip(get_output(self.model, self.sym_x), epsilon, 1)
        loss_cc = aggregate(categorical_crossentropy(y_train, self.sym_t), mode='mean')
        loss_train_acc = categorical_accuracy(y_train, self.sym_t).mean()

        y = T.clip(get_output(self.model, self.sym_x, deterministic=True), epsilon, 1)
        loss_eval = aggregate(categorical_crossentropy(y, self.sym_t), mode='mean')
        loss_acc = categorical_accuracy(y, self.sym_t).mean()

        all_params = get_all_params(self.model, trainable=True)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        grads = T.grad(loss_cc, all_params)
        grads = [T.clip(g, -5, 5) for g in grads]
        updates = rmsprop(grads, all_params, self.sym_lr, sym_beta1, sym_beta2)

        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        f_train = theano.function(
            inputs, [loss_cc, loss_train_acc],
            updates=updates,
            givens={
                self.sym_x: self.sh_train_x[self.batch_slice],
                self.sym_t: self.sh_train_t[self.batch_slice],
            },
        )

        f_test = theano.function(
            [self.sym_index, self.sym_batchsize], [loss_eval, loss_acc],
            givens={
                self.sym_x: self.sh_test_x[self.batch_slice],
                self.sym_t: self.sh_test_t[self.batch_slice],
            },
        )

        f_validate = None
        if validation_set is not None:
            f_validate = theano.function(
                [self.sym_index, self.sym_batchsize], [loss_eval, loss_acc],
                givens={
                    self.sym_x: self.sh_valid_x[self.batch_slice],
                    self.sym_t: self.sh_valid_t[self.batch_slice],
                },
            )

        self.train_args['inputs']['batchsize'] = 128
        self.train_args['inputs']['learningrate'] = 1e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        self.train_args['outputs']['loss_cc'] = '%0.6f'
        self.train_args['outputs']['loss_train_acc'] = '%0.6f'

        self.test_args['inputs']['batchsize'] = 128
        self.test_args['outputs']['loss_eval'] = '%0.6f'
        self.test_args['outputs']['loss_acc'] = '%0.6f'

        self.validate_args['inputs']['batchsize'] = 128
        # self.validate_args['outputs']['loss_eval'] = '%0.6f'
        # self.validate_args['outputs']['loss_acc'] = '%0.6f'

        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def model_info(self):
        return self.log

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)

