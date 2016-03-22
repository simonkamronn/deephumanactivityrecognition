import datetime
import time
from os import rmdir

import matplotlib.pyplot as plt
import numpy as np
from lasagne.layers import get_all_layers, get_output_shape
from lasagne.nonlinearities import rectify, softmax
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from data_preparation.load_data import LoadHAR, ACTIVITY_MAP
from lasagne_extensions.confusionmatrix import ConfusionMatrix
from models.resnet import ResNet
from training.train import TrainModel
from utils import env_paths as paths
from har_utils import one_hot

def main():
    add_pitch, add_roll, add_filter = False, False, True
    n_samples, step = 200, 200
    load_data = LoadHAR(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                        n_samples=n_samples, step=step)
    batch_size = 64

    # Define datasets and load iteratively
    datasets = [load_data.idash, load_data.wisdm1, load_data.uci_mhealth, load_data.uci_hapt]
    X, y, name, users = datasets[0]()
    users = ['%s_%02d' % (name, user) for user in users]
    for dataset in datasets[1:]:
        X_tmp, y_tmp, name_tmp, users_tmp = dataset()
        X = np.concatenate((X, X_tmp))
        y = np.concatenate((y, y_tmp))
        for user in users_tmp:
            users.append('%s_%02d' % (name_tmp, user))
        name += '_' + name_tmp
    users = np.array(users)

    print('Users: %d' % len(np.unique(users)))
    print(X.shape)

    n_windows, sequence_length, n_features = X.shape
    y = one_hot(y, n_classes=len(ACTIVITY_MAP))
    n_classes = y.shape[-1]

    # Create a time-string for our cv run
    d = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'))
    cv = LeaveOneLabelOut(users)

    user_idx = 0
    user_names = np.unique(users)
    user = None
    if user is not None:
        train_idx = users != user
        test_idx = users == user
        cv = ((train_idx, test_idx), )

    for train_index, test_index in cv:
        user = user_names[user_idx]
        user_idx += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale data using training data
        scaler = StandardScaler().fit(X_train.reshape((-1, n_features)))
        n_windows = X_train.shape[0]
        X_train = scaler.transform(X_train.reshape((-1, n_features))).reshape((n_windows, sequence_length, n_features))
        n_windows = X_test.shape[0]
        X_test = scaler.transform(X_test.reshape((-1, n_features))).reshape((n_windows, sequence_length, n_features))

        print('Xtrain mean: %f\tstd: %f' % (X_train.mean(), X_train.std()))
        print('Xtest mean: %f\tstd: %f' % (X_test.mean(), X_test.std()))
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)
        valid_set = test_set

        n_train = train_set[0].shape[0]
        n_test = test_set[0].shape[0]

        n_test_batches = 1
        n_valid_batches = None
        batch_size = n_test
        n_train_batches = n_train//batch_size
        print("n_train_batches: %d, n_test_batches: %d" % (n_train_batches, n_test_batches))

        model = ResNet(n_in=(sequence_length, n_features),
                       n_filters=[32, 32, 64, 64],
                       pool_sizes=[2, 1, 2, 1],
                       n_hidden=[512],
                       conv_dropout=0.5,
                       dropout=0.5,
                       n_out=n_classes,
                       trans_func=rectify,
                       out_func=softmax,
                       batch_size=batch_size,
                       batch_norm=True)

        if len(cv) > 1:
            # Generate root path and edit
            root_path = model.get_root_path()
            model.root_path = "%s_cv_%s_%s" % (root_path, d, user)
            paths.path_exists(model.root_path)
            rmdir(root_path)

        # Build model
        f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                              test_set,
                                                                                              None)
        train_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['learningrate'] = 0.001
        train_args['inputs']['beta1'] = 0.9
        train_args['inputs']['beta2'] = 1e-6

        test_args['inputs']['batchsize'] = batch_size
        validate_args['inputs']['batchsize'] = batch_size

        # Define confusion matrix
        cfm = ConfusionMatrix(n_classes=n_classes, class_names=ACTIVITY_MAP.values())
        print(n_classes, len(ACTIVITY_MAP.values()))

        def f_custom(model, path):
            mean_evals = model.get_output(X_test).eval()
            t_class = np.argmax(y_test, axis=1)
            y_class = np.argmax(mean_evals, axis=1)
            # cfm.batchAdd(t_class, y_class)
            # print(cfm)

            cm = confusion_matrix(t_class, y_class)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.savefig(path)

        train = TrainModel(model=model,
                           anneal_lr=0.75,
                           anneal_lr_freq=100,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=f_custom)
        train.pickle = False
        train.add_initial_training_notes("Standardizing data after adding features\
                                          \nUsing striding instead of pooling")
        train.write_to_logger("Dataset: %s" % name)
        train.write_to_logger("LOO user: %s" % user)
        train.write_to_logger("Training samples: %d" % n_train)
        train.write_to_logger("Test samples: %d" % n_test)
        train.write_to_logger("Sequence length: %d" % sequence_length)
        train.write_to_logger("Step: %d" % step)
        train.write_to_logger("Shuffle: %s" % False)
        train.write_to_logger("Add pitch: %s\nAdd roll: %s" % (add_pitch, add_roll))
        train.write_to_logger("Add filter separated signals: %s" % add_filter)
        train.write_to_logger("Transfer function: %s" % model.transf)
        train.write_to_logger("Network Architecture ---------------")
        for layer in get_all_layers(model.model):
            # print(layer.name, ": ", get_output_shape(layer))
            train.write_to_logger(layer.name + ": " + str(get_output_shape(layer)))

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
