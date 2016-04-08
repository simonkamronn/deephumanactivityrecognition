import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from .base import Model
from lasagne_extensions.nonlinearities import rectify, softmax, sigmoid
from lasagne.objectives import aggregate, squared_error
from lasagne.layers import *
from lasagne import init
from lasagne_extensions.updates import adam, rmsprop
from lasagne_extensions.layers import TiedDropoutLayer
from lasagne_extensions.layers.batch_norm import batch_norm as bn

CONST_FORGET_B = 1.
GRAD_CLIP = 5


class FCAE(Model):
    def __init__(self, n_in, n_hidden, n_out, filters, pool_sizes, stats=2, conv_stride=1,
                 trans_func=rectify, conv_dropout=0.0):
        super(FCAE, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.log = ""

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length, n_features))
        input_layer = self.l_in
        print("Input shape", get_output_shape(input_layer))

        # Reshape
        layer = ReshapeLayer(input_layer, (-1, 1, sequence_length, n_features))

        # Add input noise
        # self.log += "\nAdding noise layer: 0.05"
        # gaus_noise = GaussianNoiseLayer(reshp, sigma=0.05)

        ret = {}
        n_filters = len(filters)
        for idx, num_filters in enumerate(filters):
            ret['conv%d' % (idx+1)] = layer = bn(Conv2DLayer(layer, num_filters=num_filters, filter_size=(3, 1), pad='full'))
            if idx < n_filters - 1:
                ret['pool%d' % (idx+1)] = layer = MaxPool2DLayer(layer, pool_size=(2, 1))

        ret['enc'] = layer = GlobalPoolLayer(layer)
        print("Encoding layer shape", layer.output_shape)
        ret['ph1'] = layer = NonlinearityLayer(layer, nonlinearity=None)
        ret['ph2'] = layer = NonlinearityLayer(layer, nonlinearity=None)
        ret['unenc'] = layer = bn(InverseLayer(layer, ret['enc']))

        for idx, num_filters in enumerate(filters[::-1][1:]):
            ret['deconv%d' % (n_filters - idx)] = layer = bn(Conv2DLayer(layer, num_filters=num_filters, filter_size=(3, 1)))
            ret['depool%d' % (n_filters - idx - 1)] = layer = InverseLayer(layer, ret['pool%d' % (n_filters - idx - 1)])

        ret['deconv0'] = layer = Conv2DLayer(layer, num_filters=1, filter_size=(3, 1), nonlinearity=None)
        ret['output'] = layer = ReshapeLayer(layer, (-1, sequence_length, n_features))
        print("FCAE out shape", get_output_shape(layer))

        self.model = ret['output']
        self.model_params = get_all_params(self.model)
        self.sym_x = T.tensor3('x')

    def build_model(self, train_set, test_set, validation_set=None):
        super(FCAE, self).build_model(train_set, test_set, validation_set)

        y_train = get_output(self.model, self.sym_x)
        loss = aggregate(squared_error(y_train, self.sym_x), mode='mean')
        # loss += + 1e-4 * lasagne.regularization.regularize_network_params(self.model, lasagne.regularization.l2)

        y_test = get_output(self.model, self.sym_x, deterministic=True)
        loss_test = aggregate(squared_error(y_test, self.sym_x), mode='mean')

        all_params = get_all_params(self.model, trainable=True)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        grads = T.grad(loss, all_params)
        ngrads = lasagne.updates.total_norm_constraint(grads, 5)
        cgrads = [T.clip(g, -5, 5) for g in ngrads]
        updates = rmsprop(cgrads, all_params, self.sym_lr, sym_beta1, sym_beta2)

        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        f_train = theano.function(
            inputs, [loss],
            updates=updates,
            givens={
                self.sym_x: self.sh_train_x[self.batch_slice],
            },
        )

        f_test = theano.function(
            [self.sym_index, self.sym_batchsize], [loss_test],
            givens={
                self.sym_x: self.sh_test_x[self.batch_slice],
            },
            on_unused_input='ignore',
        )

        f_ae = None
        # f_ae = theano.function(
        #     [self.sym_batchsize], [y_test],
        #     givens={
        #         self.sym_x: self.sh_valid_x,
        #     },
        #     on_unused_input='ignore',
        # )

        self.train_args['inputs']['batchsize'] = 128
        self.train_args['inputs']['learningrate'] = 1e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 1e-6
        self.train_args['outputs']['loss'] = '%0.6f'

        self.test_args['inputs']['batchsize'] = 128
        self.test_args['outputs']['loss_test'] = '%0.6f'

        # self.validate_args['inputs']['batchsize'] = 128
        # self.validate_args['outputs']['loss_eval'] = '%0.6f'
        # self.validate_args['outputs']['loss_acc'] = '%0.6f'
        return f_train, f_test, f_ae, self.train_args, self.test_args, self.validate_args

    def model_info(self):
        return self.log

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)
