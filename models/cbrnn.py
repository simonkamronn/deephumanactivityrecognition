import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax, tanh
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne.layers import *
from lasagne_extensions.updates import adam, rmsprop
CONST_FORGET_B = 1.
GRAD_CLIP = 5


class conv_BRNN(Model):
    def __init__(self, n_in, n_hidden, n_out, n_filters, filter_sizes, pool_sizes, downsample=1,
                 grad_clip=GRAD_CLIP, peepholes=False, trans_func=rectify, out_func=softmax, batch_size=128,
                 dropout_probability=0.0, factor=8, conv_dropout=0.0):
        super(conv_BRNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(batch_size, sequence_length, n_features))
        l_prev = self.l_in
        print("Input shape", get_output_shape(l_prev))

        # Reshape
        batch_size *= factor
        sequence_length /= factor
        l_prev = ReshapeLayer(l_prev, (batch_size, 1, sequence_length, n_features))
        l_prev = DimshuffleLayer(l_prev, (0, 3, 2, 1))

        # Adding convolutional layers
        print("Conv input shape", get_output_shape(l_prev))
        for n_filter, filter_size, pool_size in zip(n_filters, filter_sizes, pool_sizes):
            self.log += "\nAdding 2D conv layer: %d x %d" % (n_filter, filter_size)
            self.log += "\nStride: %d" % pool_size
            l_prev = Conv2DLayer(l_prev,
                                 num_filters=n_filter,
                                 filter_size=(filter_size, 1),
                                 pad="same",
                                 nonlinearity=self.transf,
                                 stride=(pool_size, 1))
            if pool_size > 3:
                self.log += "\nAdding max pooling layer: %d" % pool_size
                l_prev = MaxPool2DLayer(l_prev, pool_size=(pool_size, 1))
            if conv_dropout:
                l_prev = DropoutLayer(l_prev, p=conv_dropout)
                self.log += "\nAdding dropout: %.2f" % conv_dropout
        print("Conv out shape", get_output_shape(l_prev))

        # Reshape for LSTM
        batch_size /= factor
        self.log += "\nGlobal Pooling: average"
        l_prev = GlobalPoolLayer(l_prev, pool_function=T.mean)
        l_prev = ReshapeLayer(l_prev, (batch_size, factor, -1))

        # Add BLSTM layers
        print("LSTM input shape", get_output_shape(l_prev))
        for n_hid in n_hidden:
            self.log += "\nAdding BLSTM layer with %d units" % n_hid
            l_forward = LSTMLayer(
                l_prev,
                num_units=n_hid/2,
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.GlorotUniform(),
                    W_hid=lasagne.init.GlorotUniform()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=tanh
            )
            l_backward = LSTMLayer(
                l_prev,
                num_units=n_hid/2,
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.GlorotUniform(),
                    W_hid=lasagne.init.GlorotUniform()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=tanh,
                backwards=True
            )
            print("LSTM forward shape", get_output_shape(l_prev))

            l_prev = ConcatLayer(
                [l_forward, l_backward],
                axis=2
            )
            print("LSTM concat shape", get_output_shape(l_prev))

        l_prev = ReshapeLayer(l_prev, (batch_size*factor, -1))
        # l_prev = DenseLayer(l_prev, num_units=512, nonlinearity=trans_func)
        # self.log += "\nAdding dense layer with %d units" % 512

        if dropout_probability:
            self.log += "\nAdding output dropout with probability %.2f" % dropout_probability
            l_prev = DropoutLayer(l_prev, p=dropout_probability)

        l_prev = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model = ReshapeLayer(l_prev, (batch_size, factor, n_out))
        print("Output shape", get_output_shape(self.model))

        self.model_params = get_all_params(self.model)
        self.sym_x = T.tensor3('x')
        self.sym_t = T.tensor3('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(conv_BRNN, self).build_model(train_set, test_set, validation_set)

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
        grads = T.grad(loss_cc, all_params)
        grads = [T.clip(g, -GRAD_CLIP, GRAD_CLIP) for g in grads]
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