#import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpuo')
from models.conv_brnn import conv_BRNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
import load_data as ld
import numpy as np


def main():
    add_pitch, add_roll, add_filter = True, True, True
    step = 200
    shuffle = True
    batch_size = 64
    (train_set, test_set, valid_set, (sequence_length, n_features, n_classes)), name = \
        ld.LoadHAR().uci_hapt(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                              n_samples=200, step=step, shuffle=shuffle)

    # The data is structured as (samples, sequence, features) but to properly use the convolutional RNN we need a longer
    # time
    factor = 5
    sequence_length *= factor

    n_train = train_set[0].shape[0]//factor
    print("Resizing train set from %d to %d" % (train_set[0].shape[0], n_train*factor))
    train_set = (np.reshape(train_set[0][:factor*n_train], (n_train, sequence_length, n_features)),
                 np.reshape(train_set[1][:factor*n_train], (n_train, factor, n_classes)))

    n_test = test_set[0].shape[0]//factor
    print("Resizing test set from %d to %d" % (test_set[0].shape[0], n_test*factor))
    test_set = (np.reshape(test_set[0][:factor*n_test], (n_test, sequence_length, n_features)),
                np.reshape(test_set[1][:factor*n_test], (n_test, factor, n_classes)))

    valid_set = test_set

    n_train = train_set[0].shape[0]
    n_test = test_set[0].shape[0]
    n_valid = valid_set[0].shape[0]

    n_train_batches = n_train//batch_size
    n_test_batches = n_test//batch_size
    n_valid_batches = n_valid//batch_size

    print("n_train_batches: %d, n_test_batches: %d, n_valid_batches: %d"
          % (n_train_batches, n_test_batches, n_valid_batches))

    n_conv = 5
    model = conv_BRNN(n_in=(sequence_length, n_features),
                      n_filters=[64]*n_conv,
                      filter_sizes=[3]*n_conv,
                      pool_sizes=[2]*n_conv,
                      n_hidden=[100],
                      conv_dropout=0.1,
                      dropout_probability=0.5,
                      n_out=n_classes,
                      downsample=1,
                      trans_func=rectify,
                      out_func=softmax,
                      batch_size=batch_size,
                      factor=factor)

    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                          test_set,
                                                                                          valid_set)
    train_args['inputs']['batchsize'] = batch_size
    train_args['inputs']['learningrate'] = 0.004
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 1e-6

    if n_test_batches == 1:
        test_args['inputs']['index'] = 1
    test_args['inputs']['batchsize'] = batch_size

    if n_valid_batches == 1:
        validate_args['inputs']['index'] = 1
    validate_args['inputs']['batchsize'] = batch_size

    model.log += "\nDataset: %s" % name
    model.log += "\nTraining samples: %d" % n_train
    model.log += "\nTest samples: %d" % n_test
    model.log += "\nSequence length: %d" % (sequence_length/factor)
    model.log += "\nTime steps: %d" % factor
    model.log += "\nStep: %d" % step
    model.log += "\nShuffle %s" % shuffle
    model.log += "\nAdd pitch: %s\nAdd roll: %s" % (add_pitch, add_roll)
    model.log += "\nAdd filter separated signals: %s" % add_filter
    model.log += "\nTransfer function: %s" % model.transf
    train = TrainModel(model=model,
                       anneal_lr=0.9,
                       anneal_lr_freq=50,
                       output_freq=1,
                       pickle_f_custom_freq=100,
                       f_custom_eval=None)
    train.pickle = True
    train.add_initial_training_notes("")
    train.train_model(f_train=f_train, train_args=train_args,
                      f_test=f_test, test_args=test_args,
                      f_validate=f_validate, validation_args=validate_args,
                      n_train_batches=n_train_batches,
                      n_test_batches=n_test_batches,
                      n_valid_batches=n_valid_batches,
                      n_epochs=1000)

if __name__ == "__main__":
    main()
