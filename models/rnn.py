import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, get_output_shape, LSTMLayer, Gate, ConcatLayer, DenseLayer, \
    DropoutLayer, InputLayer, SliceLayer, get_all_params
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne_extensions.updates import adam, rmsprop
CONST_FORGET_B = 1.
GRAD_CLIP = 5


class RNN(Model):
    def __init__(self, n_in, n_hidden, n_out, ccf=False, grad_clip=GRAD_CLIP, peepholes=False,
                 trans_func=rectify, out_func=softmax, dropout_probability=0.0):
        super(RNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        # Define model using lasagne framework
        dropout = True if not dropout_probability == 0.0 else False

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length, n_features))
        l_prev = self.l_in

        if ccf:
            self.log += "\nAdding cross-channel feature layer"
            l_prev = lasagne.layers.ReshapeLayer(l_prev, (-1, 1, sequence_length, n_features))
            l_prev = lasagne.layers.Conv2DLayer(l_prev,
                                                num_filters=n_features*3,
                                                filter_size=(1, n_features),
                                                nonlinearity=None)
            n_features *= 3
            l_prev = lasagne.layers.ReshapeLayer(l_prev, (-1, n_features, sequence_length))
            l_prev = lasagne.layers.DimshuffleLayer(l_prev, (0, 2, 1))

        print("LSTM input shape", get_output_shape(l_prev))
        for n_hid in n_hidden:
            self.log += "\nAdding BLSTM layer with %d units" % n_hid
            l_forward = LSTMLayer(
                l_prev,
                num_units=n_hid/2,
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.HeUniform(),
                    W_hid=lasagne.init.HeUniform()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=lasagne.nonlinearities.tanh
            )
            l_backward = LSTMLayer(
                l_prev,
                num_units=n_hid/2,
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.HeUniform(),
                    W_hid=lasagne.init.HeUniform()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=lasagne.nonlinearities.tanh,
                backwards=True
            )
            print("LSTM forward shape", get_output_shape(l_prev))

            l_prev = ConcatLayer(
                [l_forward, l_backward],
                axis=2
            )
            print("LSTM concat shape", get_output_shape(l_prev))

        # Slicing out the last units for classification
        l_forward_slice = SliceLayer(l_forward, -1, 1)
        l_backward_slice = SliceLayer(l_backward, 0, 1)

        print("LSTM forward slice shape", get_output_shape(l_prev))
        l_prev = ConcatLayer(
            [l_forward_slice, l_backward_slice],
            axis=1
        )

        if dropout:
            self.log += "\nAdding output dropout with probability %.2f" % dropout_probability
            l_prev = DropoutLayer(l_prev, p=dropout_probability)

        print("Output input shape", get_output_shape(l_prev))
        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None):
        super(RNN, self).build_model(train_set, test_set, validation_set)

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

        self.train_args['inputs']['batchsize'] = 64
        self.train_args['inputs']['learningrate'] = 1e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 1e-6
        self.train_args['outputs']['loss_cc'] = '%0.6f'
        self.train_args['outputs']['loss_train_acc'] = '%0.6f'

        self.test_args['inputs']['batchsize'] = 64
        self.test_args['outputs']['loss_eval'] = '%0.6f'
        self.test_args['outputs']['loss_acc'] = '%0.6f'

        self.validate_args['inputs']['batchsize'] = 64
        # self.validate_args['outputs']['loss_eval'] = '%0.6f'
        # self.validate_args['outputs']['loss_acc'] = '%0.6f'
        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def model_info(self):
        return self.log

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)
