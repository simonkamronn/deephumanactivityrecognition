__author__ = 'Simon'
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu2')

import theano.tensor as T
from models.rnn import RNN
from training.train import TrainModel
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import rectify, softmax
from lasagne.updates import adam, nesterov_momentum, rmsprop
import load_data as ld


def run_cnn():
    train_set, test_set, valid_set, (sequence_length, n_features, n_classes) = ld.LoadHAR().uci_har_v1(True, True)
    model = RNN(n_in=(sequence_length, n_features),
                n_hidden=(40, 40),
                n_out=n_classes,
                grad_clip= 5,
                downsample=0,
                ccf=False,
                trans_func=rectify,
                out_func=softmax,
                batch_size=100,
                dropout_probability=0.0)

    train = TrainModel(model,
                       train_set, test_set, valid_set,
                       loss=categorical_crossentropy,
                       update=rmsprop,
                       update_args=(.001, .95),
                       eval_freq=10,
                       pickle=True,
                       custom_eval_func=None)
    train.train_model(1000)

if __name__ == "__main__":
    run_cnn()
