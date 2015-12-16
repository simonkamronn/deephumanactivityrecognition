from models.cnn import CNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify, leaky_rectify
import load_data as ld
from sklearn.cross_validation import LeaveOneLabelOut
import numpy as np
from utils import env_paths as paths
import cPickle as pkl

def run_cnn():
    add_pitch, add_roll, add_filter = False, False, True
    n_samples, step = 200, 100
    shuffle = False
    batch_size = 32
    (train_set, test_set, valid_set, (sequence_length, n_features, n_classes)), name, users = \
        ld.LoadHAR().uci_hapt(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                              n_samples=n_samples, step=step, shuffle=shuffle)

    X = np.concatenate((train_set[0], test_set[0]), axis=0)
    y = np.concatenate((train_set[1], test_set[1]), axis=0)

    model = CNN(n_in=(sequence_length, n_features),
                n_filters=[32, 32, 64, 64, 64, 64],
                filter_sizes=[3, 3, 3, 3, 3, 3],
                pool_sizes=[0, 2, 0, 2, 2, 2],
                conv_dropout=0.2,
                n_hidden=[512],
                dropout_probability=0.5,
                n_out=n_classes,
                downsample=0,
                ccf=False,
                trans_func=rectify,
                out_func=softmax,
                batch_size=batch_size,
                batch_norm=False)
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

        n_train = train_set[0].shape[0]
        n_test = test_set[0].shape[0]

        n_train_batches = n_train//batch_size
        n_test_batches = n_test//batch_size
        n_valid_batches = n_test//batch_size

        print("n_train_batches: %d, n_test_batches: %d" % (n_train_batches, n_test_batches))


        model.model_params = base_params
        f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                              test_set,
                                                                                              valid_set)
        train_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['learningrate'] = 0.004
        train_args['inputs']['beta1'] = 0.9
        train_args['inputs']['beta2'] = 1e-6

        test_args['inputs']['batchsize'] = batch_size
        validate_args['inputs']['batchsize'] = batch_size

        train = TrainModel(model=model,
                           anneal_lr=0.75,
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
        train.write_to_logger("Sequence length: %d" % sequence_length)
        train.write_to_logger("Step: %d" % step)
        train.write_to_logger("Shuffle: %s" % shuffle)
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
    run_cnn()
