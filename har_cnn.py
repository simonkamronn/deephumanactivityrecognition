__author__ = 'Simon'
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')

from models.cnn import CNN
from training.train import TrainModel
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import rectify, softmax
from lasagne.updates import adam
import load_data as ld


def run_cnn():
    data = ld.LoadHAR().uci_har_v1(True, True)
    model = CNN((128, 5), [64, 64, 64, 64], [5, 5, 3, 3], 6, downsample=2, sum_channels=False, ccf=False,
                trans_func=rectify, out_func=softmax, batch_size=100, dropout_probability=0.0, pool_sizes=[0, 2, 0, 2])
    train = TrainModel(model, *data, loss=categorical_crossentropy, update=adam, update_args=(0.0005,), eval_freq=10,
                       pickle=True, custom_eval_func=None)
    train.train_model(200)

if __name__ == "__main__":
    run_cnn()
