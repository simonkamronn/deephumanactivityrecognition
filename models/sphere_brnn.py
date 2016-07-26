import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
from .base import Model
from lasagne_extensions.nonlinearities import rectify, softmax
from lasagne.layers import get_output, get_output_shape, LSTMLayer, Gate, ConcatLayer, DenseLayer, \
    DropoutLayer, InputLayer, SliceLayer, get_all_params, FeaturePoolLayer, ReshapeLayer, batch_norm, \
    BatchNormLayer, NonlinearityLayer
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne_extensions.updates import adam, rmsprop
from lasagne import init
CONST_FORGET_B = 1.
GRAD_CLIP = 5


class BRNN(Model):
    def __init__(self, n_in, n_hidden, n_out, n_enc, enc_values, freeze_encoder=True, grad_clip=GRAD_CLIP,
                 peepholes=False, fs=20, bn=False,
                 trans_func=rectify, out_func=softmax, dropout=0.0, bl_dropout=0.0, slicers=None):
        super(BRNN, self).__init__(n_in, n_hidden, n_out, trans_func)
        self.outf = out_func
        self.log = ""

        def _lstm_layer(incoming, n_hid, backwards=False, return_final=False, bn=False):
            lstm = LSTMLayer(
                incoming,
                num_units=int(n_hid),
                grad_clipping=grad_clip,
                peepholes=peepholes,
                ingate=Gate(
                    W_in=lasagne.init.GlorotNormal(),
                    W_hid=lasagne.init.Orthogonal()
                ),
                forgetgate=Gate(
                    b=lasagne.init.Constant(CONST_FORGET_B)
                ),
                nonlinearity=lasagne.nonlinearities.tanh,
                backwards=backwards,
                only_return_final=return_final
            )
            if bn:
                self.log += "\nAdding batchnorm"
                lstm = batch_norm(lstm)

            return lstm

        def _lstm_module(incoming, n_hidden, dropout, bn):
            l_prev = incoming
            for i, n_hid in enumerate(n_hidden):
                return_final = False if (len(n_hidden)-1 > i) else True
                l_prev = _lstm_layer(l_prev, n_hid, return_final=return_final, bn=bn)

                if len(n_hidden) - 1 > i:
                    if bl_dropout:
                        self.log += "\nAdding lstm dropout with probability %.2f" % dropout
                        l_prev = DropoutLayer(l_prev, p=dropout)

            if dropout:
                self.log += "\nAdding lstm dropout with probability %.2f" % dropout
                l_prev = DropoutLayer(l_prev, p=dropout)

            return l_prev

        def _blstm_layer(incoming, n_hid, return_final=False):
            self.log += "\nAdding BLSTM layer with %d units" % n_hid
            l_forward = _lstm_layer(incoming, n_hid/2, False, return_final)
            l_backward = _lstm_layer(incoming, n_hid/2, True, return_final)

            out = ConcatLayer(
                [l_forward, l_backward],
                axis=2
            )
            return out, l_forward, l_backward

        def _blstm_module(incoming, n_hidden, bl_dropout, bn):
            l_prev = incoming
            for i, n_hid in enumerate(n_hidden):
                l_prev, l_forward, l_backward = _blstm_layer(l_prev, n_hid)

                if len(n_hidden) - 1 > i:
                    if bn:
                        self.log += "\nAdding batchnorm"
                        l_prev = batch_norm(l_prev)
                    if bl_dropout > .0:
                        self.log += "\nAdding between layer dropout: %.2f" % dropout
                        l_prev = DropoutLayer(l_prev, p=bl_dropout)

            # Slicing out the last units for classification
            l_forward_slice = SliceLayer(l_forward, -1, 1)
            l_backward_slice = SliceLayer(l_backward, 0, 1)
            l_prev = ConcatLayer(
                [l_forward_slice, l_backward_slice],
                axis=1
            )

            return l_prev

        def mean_layer(incoming, pool_size, axis=-1):
            l_out = FeaturePoolLayer(incoming, pool_size=pool_size, axis=axis, pool_function=T.mean)
            return SliceLayer(l_out, indices=0, axis=axis)

        def slice_mean_layer(incoming, pool_size, axis=-1):
            l_slice = SliceLayer(incoming, -1)
            return FeaturePoolLayer(l_slice, pool_size=pool_size, axis=axis, pool_function=T.mean)

        def rnn_encoder(l_x_in, enc_rnn):
            # Decide Glorot initializaiton of weights.
            init_w = 1e-3
            hid_w = "relu"

            # Assist methods for collecting the layers
            def dense_layer(layer_in, n, dist_w=init.GlorotNormal, dist_b=init.Normal):
                dense = DenseLayer(layer_in, n, dist_w(hid_w), dist_b(init_w), nonlinearity=None)
                if bn:
                    dense = BatchNormLayer(dense)
                return NonlinearityLayer(dense, self.transf)

            def lstm_layer(input, nunits, return_final, backwards=False, name='LSTM'):
                ingate = Gate(W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01), b=init.Constant(0.0))
                forgetgate = Gate(W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01), b=init.Constant(5.0))
                cell = Gate(W_cell=None, nonlinearity=T.tanh, W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01), )
                outgate = Gate(W_in=init.Uniform(0.01), W_hid=init.Uniform(0.01), b=init.Constant(0.0))

                lstm = LSTMLayer(input, num_units=nunits, backwards=backwards,
                                 peepholes=False,
                                 ingate=ingate,
                                 forgetgate=forgetgate,
                                 cell=cell,
                                 outgate=outgate,
                                 name=name,
                                 only_return_final=return_final)
                return lstm

            # RNN encoder implementation
            l_enc_forward = lstm_layer(l_x_in, enc_rnn, return_final=True, backwards=False, name='enc_forward')
            l_enc_backward = lstm_layer(l_x_in, enc_rnn, return_final=True, backwards=True, name='enc_backward')
            l_enc_concat = ConcatLayer([l_enc_forward, l_enc_backward], axis=-1)
            l_enc = dense_layer(l_enc_concat, enc_rnn)
            return l_enc

        # Define input
        sequence_length, n_features = n_in
        self.l_in = InputLayer(shape=(None, sequence_length, n_features))
        inputs = self.l_in
        print('input shape', inputs.output_shape)

        # Reshape into 1 second windows on axis 1
        l_reshape = ReshapeLayer(inputs, (-1, fs, n_features))

        # Acceleration
        l_accel = SliceLayer(l_reshape, slicers['accel'])
        l_accel_missing = slice_mean_layer(l_accel, fs, 1)
        # l_accel_enc = _lstm_module(l_accel, n_hidden, dropout, bn)

        # Build encoder network
        l_accel_enc = rnn_encoder(l_accel, n_enc)
        print('l_accel shape', l_accel_enc.output_shape)

        # Set values of encoder network
        params = get_all_params(l_accel_enc)
        for idx, v in enumerate(enc_values):
            p = params[idx]
            if p.get_value().shape != v.shape:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" % (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

        if freeze_encoder:
            # Freeze the encoder network
            for layer in lasagne.layers.get_all_layers(l_accel_enc):
                    for param in layer.params:
                        layer.params[param].discard('trainable')

        # RSSI
        l_rssi = SliceLayer(l_reshape, slicers['rssi'])
        l_rssi = mean_layer(l_rssi, fs, 1)
        print('l_rssi shape', l_rssi.output_shape)

        # PIR
        l_pir = SliceLayer(l_reshape, slicers['pir'])
        l_pir = mean_layer(l_pir, fs, 1)
        print('l_pir shape', l_pir.output_shape)

        # Video
        l_video = SliceLayer(l_reshape, slicers['video'])
        l_video = mean_layer(l_video, fs, 1)
        print('l_video shape', l_video.output_shape)

        # Collect all embeddings
        l_concat = ConcatLayer([l_accel_enc, l_pir, l_rssi, l_video, l_accel_missing], axis=1)
        print('l_concat', l_concat.output_shape)

        # Reshape to get time steps along axis 1
        l_reshape = ReshapeLayer(l_concat, (-1, int(sequence_length//fs), n_enc+5+10+10+1))
        print('l_reshape', l_reshape.output_shape)

        # Size should now be (1, 29, feature_size)
        for n_hid in n_hidden:
            l_prev, _, _= _blstm_layer(l_reshape, n_hid, return_final=False)
            print('l_blstm', l_prev.output_shape)

            if bn:
                # Add batchnorm
                l_prev = batch_norm(l_prev)
                self.log += "\nAdding batchnorm"

            if bl_dropout > 0:
                self.log += "\nAdding dropout with probability %.2f" % bl_dropout
                l_prev = DropoutLayer(l_prev, p=dropout)

        # Reshape to process each time step individually
        l_prev = ReshapeLayer(l_prev, (-1, n_hidden[-1]))
        # l_prev = DenseLayer(l_prev, num_units=n_hidden[-1], nonlinearity=rectify)
        # if bn:
        #     l_prev = batch_norm(l_prev)
        #     self.log += "\nAdding batchnorm"
        # if dropout > 0:
        #     l_prev = DropoutLayer(l_prev, p=dropout)
        #     self.log += "\nAdding dropout"

        l_out = DenseLayer(l_prev, num_units=n_out, nonlinearity=out_func)
        self.model = ReshapeLayer(l_out, (-1, int(sequence_length//fs), n_out))
        self.model_params = get_all_params(self.model)
        print("Output shape", self.model.output_shape)

        self.sym_x = T.tensor3('x')
        self.sym_t = T.tensor3('t')

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
        grads = T.grad(train_brier, all_params)
        grads = [T.clip(g, -1, 1) for g in grads]
        updates = adam(grads, all_params, self.sym_lr, sym_beta1, sym_beta2)

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
        self.train_args['inputs']['beta2'] = 0.999  # 1e-6
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
