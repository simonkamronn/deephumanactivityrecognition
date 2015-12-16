from models.conv_brnn import conv_BRNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
import load_data as ld
import numpy as np
from sklearn.cross_validation import LeaveOneLabelOut
from utils import env_paths as paths
import cPickle as pkl

def main():
    add_pitch, add_roll, add_filter = True, True, True
    n_samples, step = 200, 50
    shuffle = False
    batch_size = 32
    (train_set, test_set, valid_set, (sequence_length, n_features, n_classes)), name, users = \
        ld.LoadHAR().uci_hapt(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                              n_samples=n_samples, step=step, shuffle=shuffle)

    # The data is structured as (samples, sequence, features) but to properly use the convolutional RNN we need a longer
    # time
    factor = 3
    sequence_length *= factor

    # Concat train and test data
    X = np.concatenate((train_set[0], test_set[0]), axis=0)
    y = np.concatenate((train_set[1], test_set[1]), axis=0)

    n_conv = 6
    model = conv_BRNN(n_in=(sequence_length, n_features),
                      n_filters=[64]*n_conv,
                      filter_sizes=[3]*n_conv,
                      pool_sizes=[0, 2]*(n_conv/2),
                      n_hidden=[100],
                      conv_dropout=0.1,
                      dropout_probability=0.5,
                      n_out=n_classes,
                      downsample=1,
                      trans_func=rectify,
                      out_func=softmax,
                      batch_size=batch_size,
                      factor=factor)
    base_params = model.model_params

    lol = LeaveOneLabelOut(users)
    user = 0
    eval_validation = np.empty((0, 2))
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

        n_train_batches = n_train//batch_size
        n_test_batches = n_test//batch_size
        n_valid_batches = n_valid//batch_size

        print("n_train_batches: %d, n_test_batches: %d, n_valid_batches: %d"
              % (n_train_batches, n_test_batches, n_valid_batches))

        model.model_params = base_params
        f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                              test_set,
                                                                                              valid_set)
        train_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['learningrate'] = 0.003
        train_args['inputs']['beta1'] = 0.9
        train_args['inputs']['beta2'] = 1e-6
        test_args['inputs']['batchsize'] = batch_size
        validate_args['inputs']['batchsize'] = batch_size

        train = TrainModel(model=model,
                           anneal_lr=0.9,
                           anneal_lr_freq=100,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=None)
        train.pickle = True
        train.add_initial_training_notes("")
        train.write_to_logger("Dataset: %s" % name)
        train.write_to_logger("LOO user: %d" % user)
        train.write_to_logger("Training samples: %d" % n_train)
        train.write_to_logger("Test samples: %d" % n_test)
        train.write_to_logger("Sequence length: %d" % (sequence_length/factor))
        train.write_to_logger("Step: %d" % step)
        train.write_to_logger("Time steps: %d" % factor)
        train.write_to_logger("Shuffle: %s" % shuffle)
        train.write_to_logger("Add pitch: %s\nAdd roll: %s" % (add_pitch, add_roll))
        train.write_to_logger("Add filter separated signals: %s" % add_filter)
        train.write_to_logger("Transfer function: %s" % model.transf)
        train.train_model(f_train=f_train, train_args=train_args,
                          f_test=f_test, test_args=test_args,
                          f_validate=f_validate, validation_args=validate_args,
                          n_train_batches=n_train_batches,
                          n_test_batches=n_test_batches,
                          n_valid_batches=n_valid_batches,
                          n_epochs=500)

        # Collect
        eval_validation = np.concatenate((eval_validation, np.max(train.eval_validation.values(), axis=0).reshape(1, 2)), axis=0)
        print(eval_validation)

        # Reset logging
        handlers = train.logger.handlers[:]
        for handler in handlers:
            handler.close()
            train.logger.removeHandler(handler)
        del train.logger

    cv_eval = paths.get_plot_evaluation_path_for_model(model.get_root_path(), "_cv.pkl")
    pkl.dump(eval_validation, open(cv_eval, "wb"))

if __name__ == "__main__":
    main()
