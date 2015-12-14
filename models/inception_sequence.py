import theano
theano.config.floatX = 'float32'
import theano.tensor as T
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, get_output_shape, DenseLayer, DropoutLayer, InputLayer, \
    ReshapeLayer, DimshuffleLayer, get_all_params, Conv2DLayer, Pool2DLayer, GlobalPoolLayer, NonlinearityLayer, BiasLayer
from lasagne_extensions.layers.batch_norm import BatchNormLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne_extensions.layers.batch_norm import batch_norm as batch_norm_layer
from lasagne_extensions.updates import adam
from inception_module import inception_module


class Inception_seq(Model):
    def __init__(self, n_in, inception_layers, n_out, pool_sizes=None, n_hidden=(512),
                 batch_size=128, trans_func=rectify, out_func=softmax, dropout_probability=0.0,
                 batch_norm=False, inception_dropout=0.0):
        super(Inception_seq, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Define model using lasagne framework
        dropout = True if not dropout_probability == 0.0 else False

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(batch_size, sequence_length, n_features), name='Input')
        l_prev = self.l_in

        # 2D Convolutional layers--------------
        l_prev = ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features), name='Reshape')
        l_prev = DimshuffleLayer(l_prev, (0, 3, 2, 1), name='Dimshuffle')

        # Init with a Conv
        self.log += "\nAdding 2D conv layer: %d x %d" % (32, 3)
        l_prev = Conv2DLayer(l_prev, num_filters=32, filter_size=(3, 1), pad='same', nonlinearity=None, b=None, name='Input Conv2D')
        l_prev = BatchNormalizeLayer(l_prev, normalize=batch_norm)

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
            l_prev = DropoutLayer(l_prev, p=inception_dropout, name='Inception dropout')

            print("Conv out shape", get_output_shape(l_prev))

        # Global pooling layer
        l_prev = GlobalPoolLayer(l_prev, pool_function=T.max, name='Global max pool')

        for n_hid in n_hidden:
            self.log += "\nAdding dense layer with %d units" % n_hid
            print("Dense input shape", get_output_shape(l_prev))
            l_prev = DenseLayer(l_prev, num_units=n_hid, nonlinearity=self.transf, name='Dense')
            if batch_norm:
                l_prev = batch_norm_layer(l_prev)
            if dropout:
                self.log += "\nAdding output dropout with probability %.2f" % dropout_probability
                l_prev = DropoutLayer(l_prev, p=dropout_probability, name='Dense dropout')

        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func, name='Output')
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(Inception_seq, self).build_model(train_set, test_set, validation_set)

        epsilon = 1e-8
        loss_cc = aggregate(categorical_crossentropy(
            T.clip(get_output(self.model, self.sym_x), epsilon, 1),
            self.sym_t
        ), mode='mean')

        y = T.clip(get_output(self.model, self.sym_x, deterministic=True), epsilon, 1)
        loss_eval = aggregate(categorical_crossentropy(y, self.sym_t), mode='mean')
        loss_acc = categorical_accuracy(y, self.sym_t).mean()

        all_params = get_all_params(self.model, trainable=True)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        updates = adam(loss_cc, all_params, self.sym_lr, sym_beta1, sym_beta2)

        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        f_train = theano.function(
            inputs, [loss_cc],
            updates=updates,
            givens={
                self.sym_x: self.sh_train_x[self.batch_slice],
                self.sym_t: self.sh_train_t[self.batch_slice],
            },
        )

        f_test = theano.function(
            [self.sym_index, self.sym_batchsize], [loss_eval],
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

        self.test_args['inputs']['batchsize'] = 128
        self.test_args['outputs']['loss_eval'] = '%0.6f'

        self.validate_args['inputs']['batchsize'] = 128
        self.validate_args['outputs']['loss_eval'] = '%0.6f'
        self.validate_args['outputs']['loss_acc'] = '%0.6f%%'
        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def model_info(self):
        return self.log

def BatchNormalizeLayer(l_prev, normalize=False, nonlinearity=rectify):
    if normalize:
        # l_prev = NormalizeLayer(l_prev, alpha='single_pass')
        # l_prev = ScaleAndShiftLayer(l_prev)
        # l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity)
        l_prev = BatchNormLayer(l_prev, nonlinearity=nonlinearity, name='Batch norm')
    else:
        l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity, name='Non-linearity')
        l_prev = BiasLayer(l_prev, name='Bias')
    return l_prev