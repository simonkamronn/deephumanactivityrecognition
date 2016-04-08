import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from .base import Model
from lasagne_extensions.nonlinearities import rectify, softmax, sigmoid, leaky_rectify
from lasagne.objectives import aggregate, squared_error
from lasagne.layers import *
from lasagne import init
from lasagne_extensions.updates import adam, rmsprop
from lasagne_extensions.layers import TiedDropoutLayer
from lasagne_extensions.layers.batch_norm import batch_norm as bn
from lasagne_extensions.updates import total_norm_constraint

CONST_FORGET_B = 1.
GRAD_CLIP = 5


class CAE(Model):
    def __init__(self, n_in, n_hidden, n_out, filters, stats=2, conv_stride=1,
                 trans_func=rectify, conv_dropout=0.0):
        super(CAE, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.log = ""
        self.sym_x = T.tensor3('x')

        # Overwrite input layer
        sequence_length, n_features = n_in
        l_x_in = InputLayer(shape=(None, sequence_length, n_features))
        print("Input shape", get_output_shape(l_x_in))

        # Reshape
        layer = ReshapeLayer(l_x_in, (-1, 1, sequence_length, n_features))

        # Add input noise
        # self.log += "\nAdding noise layer: 0.05"
        # gaus_noise = GaussianNoiseLayer(reshp, sigma=0.05)

        ret = {}
        n_filters = len(filters)
        for idx, num_filters in enumerate(filters):
            ret['conv%d' % (idx+1)] = layer = bn(Conv2DLayer(layer, num_filters=num_filters, filter_size=(3, 1), pad='full', W=init.GlorotNormal('relu'), nonlinearity=trans_func))
            if idx < n_filters - 1:
                ret['pool%d' % (idx+1)] = layer = MaxPool2DLayer(layer, pool_size=(2, 1))
            print("Convolution+pooling output", layer.output_shape)
        ret['global_pool'] = layer = GlobalPoolLayer(layer)

        print("Encoding layer input", layer.output_shape)
        # s1, s2, s3, s4 = layer.output_shape
        # ret['flatten'] = layer = ReshapeLayer(layer, (-1, s2*s3*s4))
        ret['enc'] = layer = bn(DenseLayer(layer, num_units=n_hidden, nonlinearity=trans_func))

        print("Decoding input", layer.output_shape)
        ret['hidden'] = layer = bn(DenseLayer(layer, num_units=filters[-1], nonlinearity=trans_func))
        # ret['unflatten'] = layer = ReshapeLayer(layer, (-1, filters[-1], 1, 1))

        ret['depoolglobal'] = layer = bn(InverseLayer(layer, ret['global_pool']))
        print("Global depool output", layer.output_shape)

        for idx, num_filters in enumerate(filters[::-1][1:]):
            ret['deconv%d' % (n_filters - idx)] = layer = bn(Conv2DLayer(layer, num_filters=num_filters, filter_size=(3, 1), W=init.GlorotNormal('relu'), nonlinearity=trans_func))
            ret['depool%d' % (n_filters - idx - 1)] = layer = InverseLayer(layer, ret['pool%d' % (n_filters - idx - 1)])
            print("Deconv output", layer.output_shape)

        # ret['depool1'] = layer = InverseLayer(layer, ret['pool%d' % 1])
        ret['deconv1'] = layer = Conv2DLayer(layer, num_filters=1, filter_size=(3, 1), nonlinearity=None)
        ret['output'] = layer = ReshapeLayer(layer, (-1, sequence_length, n_features))
        print("CAE out shape", get_output_shape(layer))

        self.l_x_in = l_x_in

        inputs = {l_x_in: self.sym_x}
        outputs = get_output(layer, inputs, deterministic=True)
        self.f_px = theano.function([self.sym_x], outputs, on_unused_input='warn')

        self.model = ret['output']
        self.model_params = get_all_params(self.model)
        self.trainable_model_params = get_all_params(self.model, trainable=True)

    def build_model(self, train_set, test_set, validation_set=None):
        super(CAE, self).build_model(train_set, test_set, validation_set)

        y_train = get_output(self.model, self.sym_x)
        loss = aggregate(squared_error(y_train, self.sym_x), mode='mean')
        loss += + 1e-4 * lasagne.regularization.regularize_network_params(self.model, lasagne.regularization.l2)

        y_test = get_output(self.model, self.sym_x, deterministic=True)
        loss_test = aggregate(squared_error(y_test, self.sym_x), mode='mean')

        grads_collect = T.grad(loss, self.trainable_model_params)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        clip_grad, max_norm = 1, 5
        mgrads = total_norm_constraint(grads_collect, max_norm=max_norm)
        mgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
        updates = adam(mgrads, self.trainable_model_params, self.sym_lr, sym_beta1, sym_beta2)

        # Training function
        x_batch = self.sh_train_x[self.batch_slice]

        givens = {self.sym_x: x_batch}
        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        outputs = [loss]
        f_train = theano.function(inputs=inputs, outputs=outputs, givens=givens, updates=updates)

        # Validation and test function
        givens = {self.sym_x: self.sh_test_x}
        f_test = theano.function(inputs=[], outputs=[loss_test], givens=givens)


        self.train_args['inputs']['batchsize'] = 128
        self.train_args['inputs']['learningrate'] = 1e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 1e-6
        self.train_args['outputs']['loss'] = '%0.6f'

        self.test_args['outputs']['loss_test'] = '%0.6f'

        return f_train, f_test, None, self.train_args, self.test_args, self.validate_args

    def model_info(self):
        return self.log

    def get_output(self, x):
        return self.f_px(x)
