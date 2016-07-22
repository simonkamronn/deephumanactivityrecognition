import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from .base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, get_output_shape, LSTMLayer, Gate, ConcatLayer, DenseLayer, \
    DropoutLayer, InputLayer, SliceLayer, get_all_params, FeaturePoolLayer, GlobalPoolLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne_extensions.updates import adam, rmsprop
CONST_FORGET_B = 1.
GRAD_CLIP = 5


class BRNN(Model):
    def __init__(self, n_in, n_hidden, n_out, grad_clip=GRAD_CLIP, peepholes=False,
                 trans_func=rectify, out_func=softmax, dropout=0.0, bl_dropout=0.0, slicers=None):
        super(BRNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        def _blstm_layer(incoming, n_hid, return_final=False):
            self.log += "\nAdding BLSTM layer with %d units" % n_hid
            l_forward = LSTMLayer(
                incoming,
                num_units=int(n_hid / 2),
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
                only_return_final=return_final
            )
            l_backward = LSTMLayer(
                incoming,
                num_units=int(n_hid / 2),
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
                backwards=True,
                only_return_final=return_final
            )

            out = ConcatLayer(
                [l_forward, l_backward],
                axis=2
            )
            return out, l_forward, l_backward

        def _blstm_module(incoming, n_hidden, dropout, bl_dropout):
            l_prev = incoming
            for i, n_hid in enumerate(n_hidden):
                l_prev, l_forward, l_backward = _blstm_layer(l_prev, n_hid)

                if (bl_dropout > .0) & (len(n_hidden) - 1 > i):
                    self.log += "\nAdding between layer dropout: %.2f" % dropout
                    l_prev = DropoutLayer(l_prev, p=bl_dropout)

            # Slicing out the last units for classification
            l_forward_slice = SliceLayer(l_forward, -1, 1)
            l_backward_slice = SliceLayer(l_backward, 0, 1)
            l_prev = ConcatLayer(
                [l_forward_slice, l_backward_slice],
                axis=1
            )

            if dropout:
                self.log += "\nAdding output dropout with probability %.2f" % dropout
                l_prev = DropoutLayer(l_prev, p=dropout)

            return l_prev

        # Overwrite input layer
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length, n_features))
        inputs = self.l_in

        # Acceleration
        l_accel = SliceLayer(inputs, slicers['accel'])
        l_accel = _blstm_module(l_accel, n_hidden, dropout, bl_dropout)
        print('l_accel shape', l_accel.output_shape)

        # PIR
        l_pir = SliceLayer(inputs, slicers['pir'])
        l_pir = FeaturePoolLayer(l_pir, pool_size=sequence_length, axis=1, pool_function=T.mean)
        l_pir = SliceLayer(l_pir, 1, 1)
        print('l_pir shape', l_pir.output_shape)

        # RSSI
        l_rssi = SliceLayer(inputs, slicers['rssi'])
        l_rssi = _blstm_module(l_rssi, [10], dropout, bl_dropout)
        print('l_rssi shape', l_rssi.output_shape)

        # Video
        l_video = SliceLayer(inputs, slicers['video'])
        l_video = _blstm_module(l_video, n_hidden, dropout, bl_dropout)
        print('l_video shape', l_video.output_shape)

        # Collect all embeddings
        l_concat = ConcatLayer([l_accel, l_pir, l_rssi, l_video], axis=1)

        print("Output input shape", l_concat.output_shape)
        l_prev = DenseLayer(l_concat, num_units=n_hidden[-1]*2, nonlinearity=rectify)
        self.model = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model_params = get_all_params(self.model)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.matrix('t')

    def build_model(self, train_set, test_set, validation_set=None, weights=None):
        super(BRNN, self).build_model(train_set, test_set, validation_set)

        def brier_score(given, predicted, weight_vector):
            return T.power(given - predicted, 2.0).dot(weight_vector).mean()

        epsilon = 1e-8
        y_train = T.clip(get_output(self.model, self.sym_x), epsilon, 1)
        train_brier = brier_score(y_train, self.sym_t, weights)
        train_cc = aggregate(categorical_crossentropy(y_train, self.sym_t), mode='mean')
        loss_train_acc = categorical_accuracy(y_train, self.sym_t).mean()

        y_test = T.clip(get_output(self.model, self.sym_x, deterministic=True), epsilon, 1)
        test_brier = brier_score(y_test, self.sym_t, weights)
        test_cc = aggregate(categorical_crossentropy(y_test, self.sym_t), mode='mean')
        test_acc = categorical_accuracy(y_test, self.sym_t).mean()

        all_params = get_all_params(self.model, trainable=True)
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        grads = T.grad(train_cc, all_params)
        grads = [T.clip(g, -5, 5) for g in grads]
        updates = rmsprop(grads, all_params, self.sym_lr, sym_beta1, sym_beta2)

        inputs = [self.sym_index, self.sym_batchsize, self.sym_lr, sym_beta1, sym_beta2]
        f_train = theano.function(
            inputs, [train_cc, train_brier],
            updates=updates,
            givens={
                self.sym_x: self.sh_train_x[self.batch_slice],
                self.sym_t: self.sh_train_t[self.batch_slice],
            },
        )

        f_test = theano.function(
            [], [test_cc, test_brier],
            givens={
                self.sym_x: self.sh_test_x,
                self.sym_t: self.sh_test_t,
            },
        )

        f_validate = None
        if validation_set is not None:
            f_validate = theano.function(
                [self.sym_index, self.sym_batchsize], [test_cc, test_acc],
                givens={
                    self.sym_x: self.sh_valid_x[self.batch_slice],
                    self.sym_t: self.sh_valid_t[self.batch_slice],
                },
            )

        predict = theano.function([self.sym_x], [y_test])

        self.train_args['inputs']['batchsize'] = 64
        self.train_args['inputs']['learningrate'] = 1e-3
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 1e-6
        self.train_args['outputs']['train_cc'] = '%0.4f'
        # self.train_args['outputs']['train_acc'] = '%0.4f'
        self.train_args['outputs']['train_brier'] = '%0.4f'

        # self.test_args['inputs']['batchsize'] = 64
        self.test_args['outputs']['test_cc'] = '%0.4f'
        # self.test_args['outputs']['test_acc'] = '%0.4f'
        self.test_args['outputs']['test_brier'] = '%0.4f'

        # self.validate_args['inputs']['batchsize'] = 64
        # self.validate_args['outputs']['loss_eval'] = '%0.6f'
        # self.validate_args['outputs']['test_acc'] = '%0.6f'
        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args, predict

    def model_info(self):
        return self.log

    def get_output(self, x):
        return get_output(self.model, x, deterministic=True)
