import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax, leaky_rectify
from lasagne.layers import get_output, get_output_shape, DenseLayer, DropoutLayer, InputLayer, FeaturePoolLayer, \
    ReshapeLayer, DimshuffleLayer, get_all_params, ElemwiseSumLayer, Conv2DLayer, Pool2DLayer, NonlinearityLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
# from lasagne.layers.dnn import Conv2DDNNLayer as
# from lasagne.layers.dnn import Pool2DDNNLayer as
from lasagne_extensions.layers.batchnormlayer import NormalizeLayer, ScaleAndShiftLayer
from lasagne_extensions.updates import adam


std = 0.02
p1 = 0
p2 = 0.1
p3 = 0.1
p4 = 0.1


class RCNN(Model):
    def __init__(self, n_in, n_filters, filter_sizes, n_out, pool_sizes=None, n_hidden=(), downsample=1, ccf=False,
                 trans_func=rectify, out_func=softmax, batch_size=100, dropout_probability=0.0):
        super(RCNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Define model using lasagne framework
        dropout = True if not dropout_probability == 0.0 else False

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(batch_size, sequence_length, n_features))
        l_prev = self.l_in

        # Downsample input
        if downsample > 1:
            self.log += "\nDownsampling with a factor of %d" % downsample
            l_prev = FeaturePoolLayer(l_prev, pool_size=downsample, pool_function=T.mean)
            sequence_length /= downsample

        if ccf:
            self.log += "\nAdding cross-channel feature layer"
            l_prev = ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features))
            l_prev = Conv2DLayer(l_prev,
                                 num_filters=4*n_features,
                                 filter_size=(1, n_features),
                                 nonlinearity=lasagne.nonlinearities.tanh)
            n_features *= 4
            l_prev = ReshapeLayer(l_prev, (batch_size, n_features, sequence_length))
            l_prev = DimshuffleLayer(l_prev, (0, 2, 1))

        l_prev = ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features))
        for n_filter, filter_size, pool_size in zip(n_filters, filter_sizes, pool_sizes):
            self.log += "\nAdding 2D conv layer: %d x %d" % (n_filter, filter_size)
            l_tmp = Conv2DLayer(l_prev,
                                num_filters=n_filter,
                                filter_size=(filter_size, 1),
                                nonlinearity=self.transf)
            if pool_size > 1:
                self.log += "\nAdding max pooling layer: %d" % pool_size
                l_tmp = Pool2DLayer(l_tmp, pool_size=(pool_size, 1))
            l_prev = l_tmp
            print("Conv out shape", get_output_shape(l_prev))

        l_prev = RecurrentConvLayer(l_prev, t=1)

        for n_hid in n_hidden:
            self.log += "\nAdding dense layer with %d units" % n_hid
            print("Dense input shape", get_output_shape(l_prev))
            l_prev = DenseLayer(l_prev, num_units=n_hid, nonlinearity=self.transf)
        if dropout:
            self.log += "\nAdding output dropout with probability %.2f" % dropout_probability
            l_prev = DropoutLayer(l_prev, p=dropout_probability)

        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(RCNN, self).build_model(train_set, test_set, validation_set)

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


def RecurrentConvLayer(input_layer, t=3, num_filters=64, filter_size=(7, 1)):
    input_conv = Conv2DLayer(incoming=input_layer,
                             num_filters=num_filters,
                             filter_size=(1, 1),
                             stride=(1, 1),
                             pad='same',
                             W=lasagne.init.Normal(std=std),
                             nonlinearity=None,
                             b=None)
    l_prev = BatchNormalizeLayer(input_conv)
    l_prev = Pool2DLayer(incoming=l_prev, pool_size=(2, 1))

    for _ in range(t):
        l_prev = Conv2DLayer(incoming=l_prev,
                             num_filters=num_filters,
                             filter_size=filter_size,
                             stride=(1, 1),
                             pad='same',
                             W=lasagne.init.Normal(std=std),
                             nonlinearity=None,
                             b=None)
        l_prev = ElemwiseSumLayer(incomings=[input_conv, l_prev], coeffs=1)
        l_prev = BatchNormalizeLayer(l_prev)

    l_pool = Pool2DLayer(incoming=l_prev, pool_size=(2, 1))
    l_drop = DropoutLayer(l_pool, p=p2)
    return l_drop


def BatchNormalizeLayer(incoming, nonlinearity=leaky_rectify):
    l_prev = NormalizeLayer(incoming, alpha=0.5)
    l_prev = ScaleAndShiftLayer(l_prev)
    l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity)
    return l_prev
