import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu3')
from models.rcnn import RCNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax
import load_data as ld


def main():
    add_pitch, add_roll = True, True
    batch_size = 128
    train_set, test_set, valid_set, (sequence_length, n_features, n_classes) = \
        ld.LoadHAR().uci_har_v1(add_pitch, add_roll)
    n_train = train_set[0].shape[0]
    n_test = test_set[0].shape[0]

    n_train_batches = n_train//batch_size
    n_test_batches = n_test//batch_size
    n_valid_batches = n_test//batch_size

    model = RCNN(n_in=(sequence_length, n_features),
                 n_filters=[],
                 filter_sizes=[],
                 pool_sizes=[],
                 n_hidden=[],
                 n_out=n_classes,
                 downsample=0,
                 ccf=False,
                 trans_func=rectify,
                 out_func=softmax,
                 batch_size=batch_size,
                 dropout_probability=0)

    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                          test_set,
                                                                                          valid_set)
    train_args['inputs']['batchsize'] = batch_size
    train_args['inputs']['learningrate'] = 0.0005
    train_args['inputs']['beta1'] = 0.8
    train_args['inputs']['beta2'] = 0.999

    test_args['inputs']['batchsize'] = batch_size
    validate_args['inputs']['batchsize'] = batch_size

    model.log += "\nAdd pitch: %s\nAdd roll: %s" % (add_pitch, add_roll)
    train = TrainModel(model=model,
                       anneal_lr=1.,
                       anneal_lr_freq=1.,
                       output_freq=1,
                       pickle_f_custom_freq=100,
                       f_custom_eval=None)
    train.pickle = True
    train.add_initial_training_notes("")
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_train_batches,
                      n_test_batches=n_test_batches,
                      n_valid_batches=n_valid_batches,
                      n_epochs=1000)

if __name__ == "__main__":
    main()
