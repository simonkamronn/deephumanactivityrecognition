import theano
theano.config.floatX = 'float32'
import theano.tensor as T
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, get_output_shape, DenseLayer, InputLayer, \
     get_all_params, Conv2DLayer, ConcatLayer, ReshapeLayer, DimshuffleLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne_extensions.updates import adam, rmsprop
import numpy as np


class UFCNN(Model):
    def __init__(self, n_in, n_filters, filter_size, n_out, pool_sizes=None, n_hidden=(), downsample=1
                 , batch_size=100, trans_func=rectify, out_func=softmax, dropout_probability=0.0):
        super(UFCNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        l2_mask = np.zeros((1, 1, filter_size*2+1, 1))
        l2_mask[:, :, 2::2, :] = 1
        l2_mask = l2_mask[:, :, ::-1]
        self.l2_mask = theano.shared(l2_mask.astype(theano.config.floatX), broadcastable=(True, True, False, False))

        l3_mask = np.zeros((1, 1, filter_size*4+1, 1))
        l3_mask[:, :, 4::4, :] = 1
        l3_mask = l3_mask[:, :, ::-1]
        self.l3_mask = theano.shared(l3_mask.astype(theano.config.floatX), broadcastable=(True, True, False, False))

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(batch_size, sequence_length, n_features))
        l_prev = self.l_in
        l_prev = ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features))
        l_prev = DimshuffleLayer(l_prev, (0, 3, 2, 1))

        l_h1 = Conv2DLayer(l_prev, num_filters=n_filters, filter_size=(filter_size, 1),
                           nonlinearity=self.transf, pad='same', name='h1')
        print(l_h1.W.get_value().shape)
        self.log += "\n%s:\t %s" % (l_h1.name, get_output_shape(l_h1))

        l_h2 = Conv2DLayer(l_h1, num_filters=n_filters, filter_size=(filter_size*2+1, 1),
                           nonlinearity=self.transf, pad='same', name='h2')
        print(l_h2.W.get_value().shape)
        self.log += "\n%s:\t %s" % (l_h2.name, get_output_shape(l_h2))

        l_h3 = Conv2DLayer(l_h2, num_filters=n_filters, filter_size=(filter_size*4+1, 1),
                           nonlinearity=self.transf, pad='same', name='h3')
        self.log += "\n%s:\t %s" % (l_h3.name, get_output_shape(l_h3))

        l_g3 = Conv2DLayer(l_h3, num_filters=n_filters, filter_size=(filter_size*4+1, 1),
                           nonlinearity=self.transf, pad='same', name='g3')
        self.log += "\n%s:\t %s" % (l_g3.name, get_output_shape(l_g3))

        l_h2_g3 = ConcatLayer((l_h2, l_g3), axis=1, name='l_h2_g3')
        self.log += "\n%s: %s" % (l_h2_g3.name, get_output_shape(l_h2_g3))

        l_g2 = Conv2DLayer(l_h2_g3, num_filters=n_filters, filter_size=(filter_size*2+1, 1),
                           nonlinearity=self.transf, pad='same', name='g2')
        self.log += "\n%s:\t %s" % (l_g2.name, get_output_shape(l_g2))

        l_h1_g2 = ConcatLayer((l_h1, l_g2), axis=1, name='l_h1_g2')
        self.log += "\n%s: %s" % (l_h1_g2.name, get_output_shape(l_h1_g2))
        l_g1 = Conv2DLayer(l_h1_g2, num_filters=n_filters, filter_size=(filter_size, 1),
                           nonlinearity=self.transf, pad='same', name='g1')
        self.log += "\n%s:\t %s" % (l_g1.name, get_output_shape(l_g1))

        l_prev = l_g1
        for n_hid in n_hidden:
            l_prev = DenseLayer(l_prev, num_units=n_hid, nonlinearity=self.transf)
        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(UFCNN, self).build_model(train_set, test_set, validation_set)

        epsilon = 1e-8
        loss_cc = aggregate(categorical_crossentropy(
            T.clip(get_output(self.model, self.sym_x), epsilon, 1),
            self.sym_t
        ), mode='mean')

        y = T.clip(get_output(self.model, self.sym_x, deterministic=True), epsilon, 1)
        loss_eval = aggregate(categorical_crossentropy(y, self.sym_t), mode='mean')
        loss_acc = categorical_accuracy(y, self.sym_t).mean()

        all_params = get_all_params(self.model, trainable=True)
        grads = T.grad(loss_cc, all_params)
        for idx, param in enumerate(all_params):
            param_name = param.name
            if 'h2.W' in param_name:
                print(param_name)
                grads[idx] *= self.l2_mask
            if 'h3.W' in param_name:
                print(param_name)
                grads[idx] *= self.l3_mask
            if 'g2.W' in param_name:
                print(param_name)
                grads[idx] *= self.l2_mask

        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        updates = adam(grads, all_params, self.sym_lr, sym_beta1, sym_beta2)

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