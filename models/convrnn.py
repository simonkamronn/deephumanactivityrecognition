import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne.layers import *
from lasagne import init
from lasagne_extensions.updates import adam, rmsprop
from lasagne_extensions.layers import TiedDropoutLayer
from lasagne_extensions.layers import LSTMDropoutLayer

CONST_FORGET_B = 1.
GRAD_CLIP = 5


class convRNN(Model):
    def __init__(self, n_in, n_hidden, n_out, n_filters, filter_sizes, pool_sizes, stats=2,
                 grad_clip=GRAD_CLIP, peepholes=False, trans_func=rectify, out_func=softmax, factor=8,
                 conv_dropout=0.0, rnn_in_dropout=0.0, rnn_hid_dropout=0.0, output_dropout=0.0):
        super(convRNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length, n_features))
        l_prev = self.l_in
        print("Input shape", get_output_shape(l_prev))

        # Reshape
        # batch_size *= factor
        sequence_length /= factor
        l_prev = ReshapeLayer(l_prev, (-1, 1, sequence_length+stats, n_features))
        l_prev = DimshuffleLayer(l_prev, (0, 3, 2, 1))

        # Separate into raw values and statistics
        if stats > 0:
            stats_layer = SliceLayer(l_prev, indices=slice(sequence_length, None), axis=2)
            stats_layer = ReshapeLayer(stats_layer, (-1, factor, stats*n_features))
            print('Stats layer shape', stats_layer.output_shape)
            l_prev = SliceLayer(l_prev, indices=slice(0, sequence_length), axis=2)

        # Add input noise
        # self.log += "\nAdding noise layer: 0.1"
        # l_prev = GaussianNoiseLayer(l_prev, sigma=0.1)

        # Adding convolutional layers
        print("Conv input shape", get_output_shape(l_prev))
        for n_filter, filter_size, pool_size in zip(n_filters, filter_sizes, pool_sizes):
            self.log += "\nAdding 2D conv layer: %d x %d" % (n_filter, filter_size)
            l_prev = Conv2DLayer(l_prev,
                                 num_filters=n_filter,
                                 filter_size=(filter_size, 1),
                                 pad='same',
                                 W=init.GlorotNormal('relu'),
                                 b=init.Normal(1e-3),
                                 nonlinearity=self.transf,
                                 stride=(1, 1))
            if pool_size > 1:
                self.log += "\nAdding max pooling layer: %d" % pool_size
                l_prev = MaxPool2DLayer(l_prev, pool_size=(pool_size, 1))
            if conv_dropout:
                l_prev = TiedDropoutLayer(l_prev, p=conv_dropout)
                self.log += "\nAdding dropout: %.2f" % conv_dropout
        print("Conv out shape", get_output_shape(l_prev))

        # Pool filters
        # self.log += "\nAdding global pooling"
        # l_prev = GlobalPoolLayer(l_prev, pool_function=T.mean)

        # Get output shapes from convnet
        s1, s2, s3, s4 = l_prev.output_shape

        # Reshape for LSTM
        l_prev = ReshapeLayer(l_prev, (-1, factor, s2*s3))

        # Concat with statistics
        if stats > 2:
            l_prev = ConcatLayer((l_prev, stats_layer), axis=2)

        # Add LSTM layers
        print("LSTM input shape", get_output_shape(l_prev))
        for n_hid in n_hidden:
            self.log += "\nAdding LSTM layer with: %d units" % n_hid
            self.log += "\nLSTM in dropout: %.2f" % rnn_in_dropout
            self.log += "\nLSTM hid dropout: %.2f" % rnn_hid_dropout
            l_prev = LSTMDropoutLayer(
                l_prev,
                num_units=n_hid,
                in_dropout=rnn_in_dropout,
                hid_dropout=rnn_hid_dropout,
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.HeUniform(),
                    W_hid=lasagne.init.HeUniform()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=lasagne.nonlinearities.tanh)
        print("LSTM output shape", get_output_shape(l_prev))

        # Reshape to process each timestep individually
        l_prev = ReshapeLayer(l_prev, (-1, n_hidden[-1]))
        # l_prev = DenseLayer(l_prev, num_units=512, nonlinearity=trans_func)
        # self.log += "\nAdding dense layer with %d units" % 512

        if output_dropout:
            self.log += "\nAdding output dropout with probability: %.2f" % output_dropout
            l_prev = DropoutLayer(l_prev, p=output_dropout)

        # Output
        l_prev = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model = ReshapeLayer(l_prev, (-1, factor, n_out))
        print("Output shape", get_output_shape(self.model))

        self.model_params = get_all_params(self.model)
        self.sym_x = T.tensor3('x')
        self.sym_t = T.tensor3('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(convRNN, self).build_model(train_set, test_set, validation_set)

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
            [self.sym_batchsize], [loss_eval, loss_acc],
            givens={
                self.sym_x: self.sh_test_x,
                self.sym_t: self.sh_test_t,
            },
            on_unused_input='ignore',
        )

        f_validate = None
        if validation_set is not None:
            f_validate = theano.function(
                [self.sym_batchsize], [loss_eval, loss_acc],
                givens={
                    self.sym_x: self.sh_valid_x,
                    self.sym_t: self.sh_valid_t,
                },
                on_unused_input='ignore',
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
