import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')
from models.ufcnn import UFCNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
import load_data as ld


def main():
    add_pitch, add_roll, add_filter = False, False, True
    batch_size = 128
    (train_set, test_set, valid_set, (sequence_length, n_features, n_classes)), name = \
        ld.LoadHAR().uci_hapt(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter)
    n_train = train_set[0].shape[0]
    n_test = test_set[0].shape[0]

    n_train_batches = n_train//batch_size
    n_test_batches = n_test//batch_size
    n_valid_batches = n_test//batch_size

    print("n_train_batches: %d, n_test_batches: %d" % (n_train_batches, n_test_batches))

    model = UFCNN(n_in=(sequence_length, n_features),
                  n_filters=64,
                  filter_size=7,
                  pool_sizes=0,
                  n_hidden=[512],
                  dropout_probability=0.5,
                  n_out=n_classes,
                  downsample=1,
                  trans_func=rectify,
                  out_func=softmax,
                  batch_size=batch_size)

    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                          test_set,
                                                                                          valid_set)
    train_args['inputs']['batchsize'] = batch_size
    train_args['inputs']['learningrate'] = 0.002
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999

    test_args['inputs']['batchsize'] = batch_size
    validate_args['inputs']['batchsize'] = batch_size

    model.log += "\nDataset: %s" % name
    model.log += "\nAdd pitch: %s\nAdd roll: %s" % (add_pitch, add_roll)
    model.log += "\nAdd filter separated signals: %s" % add_filter
    model.log += "\nTransfer function: %s" % model.transf.__name__
    train = TrainModel(model=model,
                       anneal_lr=0.9,
                       anneal_lr_freq=50,
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
                      n_epochs=10000)

if __name__ == "__main__":
    main()
