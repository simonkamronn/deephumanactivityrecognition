import theano
theano.config.floatX = 'float32'
import theano.tensor as T
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, DenseLayer, DropoutLayer, InputLayer, \
    ReshapeLayer, DimshuffleLayer, get_all_params, Conv2DLayer, Pool2DLayer, GlobalPoolLayer, \
    SliceLayer, ConcatLayer
from lasagne_extensions.layers.batch_norm import BatchNormLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne_extensions.updates import rmsprop, adam
from models.modules import inception_module, BatchNormalizeLayer
from lasagne_extensions.layers import TiedDropoutLayer


class Incep(Model):
    def __init__(self, n_in, inception_layers, n_out, pool_sizes=None, n_hidden=512,
                 trans_func=rectify, out_func=softmax, output_dropout=0.0, stats=0,
                 batch_norm=False, inception_dropout=0.0):
        super(Incep, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length+stats, n_features), name='Input')
        l_prev = self.l_in

        # Separate into raw values and statistics
        if stats > 0:
            stats_layer = SliceLayer(l_prev, indices=slice(sequence_length, None), axis=1)
            stats_layer = ReshapeLayer(stats_layer, (-1, stats*n_features))
            l_prev = SliceLayer(l_prev, indices=slice(0, sequence_length), axis=1)

        # 2D Convolutional layers--------------
        l_prev = ReshapeLayer(l_prev, (-1, 1, sequence_length, n_features), name='Reshape')
        l_prev = DimshuffleLayer(l_prev, (0, 3, 2, 1), name='Dimshuffle')

        # Init with a Conv
        self.log += "\nAdding 2D conv layer: %d x %d" % (32, 3)
        l_prev = Conv2DLayer(l_prev, num_filters=32, filter_size=(3, 1), pad='same', nonlinearity=None, b=None, name='Input Conv2D')
        l_prev = BatchNormalizeLayer(l_prev, normalize=batch_norm, nonlinearity=self.transf)
        l_prev = Pool2DLayer(l_prev, pool_size=(2, 1), name='Input pool')
        l_prev = TiedDropoutLayer(l_prev, p=inception_dropout, name='Input conv dropout')

        # Inception layers
        for inception_layer, pool_size in zip(inception_layers, pool_sizes):
            num_1x1, num_2x1_proj, reduce_3x1, num_3x1, reduce_5x1, num_5x1 = inception_layer
            self.log += "\nAdding inception layer: %s" % str(inception_layer)
            l_prev = inception_module(l_prev,
                                      num_1x1,
                                      num_2x1_proj,
                                      reduce_3x1,
                                      num_3x1,
                                      reduce_5x1,
                                      num_5x1,
                                      batch_norm=batch_norm)
            if pool_size > 1:
                self.log += "\nAdding max pooling layer: %d" % pool_size
                l_prev = Pool2DLayer(l_prev, pool_size=(pool_size, 1), name='Inception pool')

            self.log += "\nAdding dropout layer: %.2f" % inception_dropout
            l_prev = TiedDropoutLayer(l_prev, p=inception_dropout, name='Inception dropout')
            print("Inception out shape", l_prev.output_shape)

        # Global pooling layer
        self.log += "\nGlobal Pooling: average"
        l_prev = GlobalPoolLayer(l_prev, pool_function=T.mean, name='Global average pool')

        # Append statistics
        if stats > 0:
            l_prev = ConcatLayer((l_prev, stats_layer), axis=1)

        if n_hidden:
            self.log += "\nAdding dense layer with %d units" % n_hidden
            print("Dense input shape", l_prev.output_shape)
            l_prev = DenseLayer(l_prev, num_units=n_hidden, nonlinearity=self.transf, name='Dense')
            if batch_norm:
                l_prev = BatchNormLayer(l_prev)
            if output_dropout:
                self.log += "\nAdding output dropout with probability %.2f" % output_dropout
                l_prev = DropoutLayer(l_prev, p=output_dropout, name='Dense dropout')

        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func, name='Output')
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(Incep, self).build_model(train_set, test_set, validation_set)

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

        # self.validate_args['inputs']['batchsize'] = 128
        # self.validate_args['outputs']['loss_eval'] = '%0.6f'
        # self.validate_args['outputs']['loss_acc'] = '%0.6f'
        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def model_info(self):
        return self.log

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)