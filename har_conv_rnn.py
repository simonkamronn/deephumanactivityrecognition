from models.convrnn import convRNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
from data_preparation.load_data import LoadHAR, ACTIVITY_MAP
import numpy as np
from sklearn.cross_validation import LeaveOneLabelOut
from utils import env_paths as paths
import time
import datetime
from os import rmdir
from har_utils import one_hot


def main():
    add_pitch, add_roll, add_filter = False, False, True
    n_samples, step = 200, 200
    load_data = LoadHAR(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                        n_samples=n_samples, step=step)

    X, y, name, users = load_data.idash()

    n_windows, sequence_length, n_features = X.shape
    y = one_hot(y, n_classes=len(ACTIVITY_MAP))
    n_classes = y.shape[-1]
    # The data is structured as (samples, sequence, features) but to properly use the convolutional RNN we need a longer
    # time
    factor = 5
    sequence_length *= factor

    d = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S'))
    lol = LeaveOneLabelOut(users)
    user = 0
    for train_index, test_index in lol:
        user += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_set = (X_train, y_train)
        test_set = (X_test, y_test)

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

        n_test_batches = 1
        n_valid_batches = 1
        batch_size = n_test
        n_train_batches = n_train//batch_size

        print("n_train_batches: %d, n_test_batches: %d, n_valid_batches: %d"
              % (n_train_batches, n_test_batches, n_valid_batches))

        n_conv = 5
        model = convRNN(n_in=(sequence_length, n_features),
                        n_filters=[64]*n_conv,
                        filter_sizes=[3]*n_conv,
                        pool_sizes=[1, 2, 1, 2, 2],
                        n_hidden=[50, 50],
                        conv_dropout=0.0,
                        output_dropout=0.5,
                        n_out=n_classes,
                        trans_func=rectify,
                        out_func=softmax,
                        factor=factor)

        # Generate root path and edit
        root_path = model.get_root_path()
        model.root_path = "%s_cv_%s_%s%02d" % (root_path, d, name, user)
        paths.path_exists(model.root_path)
        rmdir(root_path)

        f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                              test_set,
                                                                                              None)
        test_args['inputs']['batchsize'] = batch_size
        validate_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['learningrate'] = 0.003
        train_args['inputs']['beta1'] = 0.9
        train_args['inputs']['beta2'] = 1e-6

        train = TrainModel(model=model,
                           anneal_lr=0.75,
                           anneal_lr_freq=50,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=None)
        train.pickle = False
        train.add_initial_training_notes("Standardizing data after adding features. Striding instead of pooling.")
        train.write_to_logger("Dataset: %s" % name)
        train.write_to_logger("LOO user: %d" % user)
        train.write_to_logger("Training samples: %d" % n_train)
        train.write_to_logger("Test samples: %d" % n_test)
        train.write_to_logger("Sequence length: %d" % (sequence_length/factor))
        train.write_to_logger("Step: %d" % step)
        train.write_to_logger("Time steps: %d" % factor)
        train.write_to_logger("Shuffle: %s" % False)
        train.write_to_logger("Add pitch: %s\nAdd roll: %s" % (add_pitch, add_roll))
        train.write_to_logger("Add filter separated signals: %s" % add_filter)
        train.write_to_logger("Transfer function: %s" % model.transf)

        train.train_model(f_train, train_args,
                          f_test, test_args,
                          f_validate, validate_args,
                          n_train_batches=n_train_batches,
                          n_test_batches=n_test_batches,
                          n_valid_batches=n_valid_batches,
                          n_epochs=500)

        # Reset logging
        handlers = train.logger.handlers[:]
        for handler in handlers:
            handler.close()
            train.logger.removeHandler(handler)
        del train.logger


if __name__ == "__main__":
    main()
