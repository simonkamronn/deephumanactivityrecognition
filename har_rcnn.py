from models.rcnn import RCNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
from lasagne.layers import get_all_layers, get_output_shape
from load_data import LoadHAR, one_hot, ACTIVITY_MAP
from sklearn.cross_validation import LeaveOneLabelOut
import numpy as np
from utils import env_paths as paths
import time
import datetime
from os import rmdir
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from lasagne_extensions.confusionmatrix import ConfusionMatrix
import matplotlib.pyplot as plt


def main():
    add_pitch, add_roll, add_filter = False, False, True
    n_samples, step = 200, 200
    load_data = LoadHAR(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                        n_samples=n_samples, step=step)
    batch_size = 64

    # Define datasets and load iteratively
    datasets = [load_data.uci_mhealth, load_data.wisdm1, load_data.uci_hapt, load_data.idash]
    X, y, name, users = datasets[0]()
    users = ['%s_%d' % (name, user) for user in users]
    for dataset in datasets[1:]:
        X_tmp, y_tmp, name_tmp, users_tmp = dataset()
        X = np.concatenate((X, X_tmp), axis=0)
        y = np.concatenate((y, y_tmp), axis=0)
        for user in users_tmp:
            users.append('%s_%d' % (name_tmp, user))
        name += '_' + name_tmp
    users = np.array(users)

    print('Users: %d' % len(np.unique(users)))
    print('X shape', X.shape)

    n_windows, sequence_length, n_features = X.shape
    y = one_hot(y, n_classes=len(ACTIVITY_MAP))
    n_classes = y.shape[-1]

    # Create a time-string for our cv run
    d = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'))
    lol = LeaveOneLabelOut(users)
    user = None

    user_idx = -1
    if user is not None:
        train_idx = users != user
        test_idx = users == user
        lol = ((train_idx, test_idx), )

    for train_index, test_index in lol:
        user_idx += 1
        user = users[user_idx]
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
        n_valid_batches = 1
        batch_size = n_test
        n_train_batches = n_train//batch_size

        print('Train set shape: ', X_train.shape)
        print('Test set shape: ', X_test.shape)
        print("n_train_batches: %d, n_test_batches: %d" % (n_train_batches, n_test_batches))

        n_conv = 1
        model = RCNN(n_in=(sequence_length, n_features),
                     n_filters=[32, 64, 128, 256],
                     filter_sizes=[3]*n_conv,
                     pool_sizes=[2]*n_conv,
                     rcl=[1, 2, 3, 4],
                     rcl_dropout=0.3,
                     n_hidden=[512],
                     dropout_probability=0.5,
                     n_out=n_classes,
                     downsample=1,
                     ccf=False,
                     trans_func=rectify,
                     out_func=softmax,
                     batch_size=batch_size,
                     batch_norm=True)

        # Generate root path and edit
        root_path = model.get_root_path()
        model.root_path = "%s_cv_%s_%s" % (root_path, d, user)
        paths.path_exists(model.root_path)
        rmdir(root_path)

        f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                              test_set,
                                                                                              test_set)
        test_args['inputs']['batchsize'] = batch_size
        validate_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['learningrate'] = 0.003
        train_args['inputs']['beta1'] = 0.9
        train_args['inputs']['beta2'] = 1e-6

        # Define confusion matrix
        cfm = ConfusionMatrix(n_classes=n_classes, class_names=ACTIVITY_MAP.values())

        def f_custom(model, path):
            mean_evals = model.get_output(X_test).eval()
            t_class = np.argmax(y_test, axis=1)
            y_class = np.argmax(mean_evals, axis=1)
            # cfm.batchAdd(t_class, y_class)
            # print(cfm)

            cm = confusion_matrix(t_class, y_class, labels=ACTIVITY_MAP.values())
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.savefig(path)

        train = TrainModel(model=model,
                           anneal_lr=0.75,
                           anneal_lr_freq=50,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=f_custom)
        train.pickle = True

        train.add_initial_training_notes("Standardizing data after adding features")
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
                          n_epochs=300)

        # Reset logging
        handlers = train.logger.handlers[:]
        for handler in handlers:
            handler.close()
            train.logger.removeHandler(handler)
        del train.logger


if __name__ == "__main__":
    main()
