import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu2')
from models.conv_rnn import conv_RNN
from training.train import TrainModel
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import rectify, softmax, tanh
from lasagne.updates import adam, nesterov_momentum, rmsprop
import load_data as ld


def run_rnn():
    add_pitch, add_roll = False, False
    train_set, test_set, valid_set, (sequence_length, n_features, n_classes) = \
        ld.LoadHAR().uci_har_v1(add_pitch, add_roll)

    model = conv_RNN(n_in=(sequence_length, n_features),
                     n_hidden=[128, 128, 128],
                     n_filters=[20],
                     filter_sizes=[3],
                     pool_sizes=[2],
                     n_out=n_classes,
                     grad_clip=5,
                     downsample=2,
                     ccf=False,
                     trans_func=None,
                     out_func=softmax,
                     batch_size=128,
                     dropout_probability=0.0)

    model_id = '20151001163030'
    model.load_model(model_id)
    model.log += '\nLoading model: %s' % model_id
    model.log += "\nAdd pitch: %s\nAdd roll: %s" % (add_pitch, add_roll)
    update_args = (.003, 0.95, 1e-6)
    model.log += '\nOptimizer: rmsprop'
    model.log += '\nUpdate args: %s' % (update_args,)
    train = TrainModel(model,
                       train_set, test_set, valid_set,
                       loss=categorical_crossentropy,
                       update=rmsprop,
                       update_args=update_args,
                       eval_freq=100,
                       pickle=True,
                       custom_eval_func=None)
    train.train_model(1000)

if __name__ == "__main__":
    run_rnn()
