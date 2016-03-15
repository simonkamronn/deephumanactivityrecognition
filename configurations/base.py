import datetime
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from sklearn.metrics import confusion_matrix
from har_utils import one_hot, expand_target, magnitude, rolling_window


class ModelConfiguration(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.name = None
        self.users = None
        self.user = None
        self.user_names = None
        self.n_classes = None
        self.d = None
        self.cv = None
        self.model = None
        self.n_features = None
        self.log = ''
        self.stats = 0
        self.f_validate = None

    def load_datasets(self, datasets, label_limit=100):
        # Load all datasets and concatenate
        X, y, name, users, stats = datasets[0]()
        users = ['%s%02d' % (name, user) for user in users]
        self.log += '\nLoading %s with %d samples' % (name, X.shape[0])

        for dataset in datasets[1:]:
            X_tmp, y_tmp, name_tmp, users_tmp, stats_tmp = dataset()
            X = np.concatenate((X, X_tmp))
            y = np.concatenate((y, y_tmp))
            stats = np.concatenate((stats, stats_tmp))
            self.log += '\nLoading %s with %d samples' % (name_tmp, X_tmp.shape[0])
            for user in users_tmp:
                users.append('%s%02d' % (name_tmp, user))
            name += '_' + name_tmp
        if len(stats) > 0:
            X = np.concatenate((X, stats), axis=1)
            self.stats = stats.shape[1]
            self.log += '\nStats: %d' % self.stats
        self.log += '\nLoaded %s with %d samples' % (name, X.shape[0])
        self.log += '\nData shape: %s' % str(X.shape)

        # Limit labels to a subset
        self.log += '\nLimiting labels to < %d' % label_limit
        limited_labels = y < label_limit
        y = y[limited_labels]
        X = X[limited_labels]
        users = np.char.asarray(users)[limited_labels]
        self.log += '\nRemaining: %d samples' % np.sum(limited_labels)

        # Compress labels
        for idx, label in enumerate(np.unique(y)):
            if not np.equal(idx, label):
                y[y == label] = idx

        # One hot encoding of labels
        y = one_hot(y)

        # Specify class variables
        self.n_classes = y.shape[-1]
        self.X, self.y, self.name, self.users = X.astype('float32'), y, name, users
        self.user_names = np.unique(self.users)
        self.n_features = X.shape[-1]

        self.d = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S'))

    def run(self, train_index, test_index, lr, n_epochs, model, train, load_data, factor=1, batch_size=None):
        x_train, x_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        n_windows, sequence_length, n_features = x_train.shape
        print('Xtrain mean: %f\tstd: %f' % (x_train.mean(), x_train.std()))
        print('Xtest mean: %f\tstd: %f' % (x_test.mean(), x_test.std()))

        def concat_sequence(x, window, step):
            return rolling_window(x.reshape(-1, x.shape[-1]).swapaxes(0, 1), window, step)\
                .swapaxes(0, 1).swapaxes(1, 2)

        # Reshape datasets to longer sequences
        if factor > 1:
            x_train = concat_sequence(x_train, factor*sequence_length, sequence_length)
            y_train = concat_sequence(y_train, factor, 1)
            x_test = concat_sequence(x_test, factor*sequence_length, sequence_length)
            y_test = concat_sequence(y_test, factor, 1)

        train_set = (x_train, y_train)
        test_set = (x_test, y_test)
        print('Train size: ', train_set[0].shape)
        print('Test size: ', test_set[0].shape)

        n_train = train_set[0].shape[0]
        n_test = test_set[0].shape[0]
        if batch_size is None:
            n_test_batches = 1
            batch_size = n_test
        else:
            n_test_batches = n_test//batch_size

        n_train_batches = n_train//batch_size
        print("n_train_batches: %d, n_test_batches: %d" % (n_train_batches, n_test_batches))

        # Build model
        f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                              test_set,
                                                                                              None)

        def f_custom(model, path):
            mean_evals = model.get_output(test_set[0]).eval()
            t_class = np.argmax(np.reshape(test_set[1], (n_test*factor, -1)), axis=1)
            y_class = np.argmax(np.reshape(mean_evals, (n_test*factor, -1)), axis=1)

            plt.clf()
            f, axarr = plt.subplots(nrows=1, ncols=2)
            axarr[0].plot(t_class, color='red')
            axarr[0].plot(y_class, linestyle='dotted')

            cm = confusion_matrix(t_class, y_class)
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
            # plt.clf()
            im = axarr[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            # plt.colorbar(im, cax=axarr[3], ticks=MultipleLocator(1), format="%.2f")
            axarr[1].set_ylabel('True')
            axarr[1].set_xlabel('Predicted')

            f.set_size_inches(18, 10)
            f.savefig(path, dpi=100)
            plt.close(f)

        if train.custom_eval_func is None:
            train.custom_eval_func = f_custom

        test_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['learningrate'] = lr
        train_args['inputs']['beta1'] = 0.9
        train_args['inputs']['beta2'] = 1e-6
        validate_args['inputs']['batchsize'] = batch_size

        train.add_initial_training_notes("Standardizing data after adding features")
        train.write_to_logger(self.log)
        train.write_to_logger("Dataset: %s" % self.name)
        train.write_to_logger("Normalizing: %s" % load_data.normalize)
        train.write_to_logger("Simple labels: %s" % load_data.simple_labels)
        train.write_to_logger("Common labels: %s" % load_data.common_labels)
        train.write_to_logger("LOO user: %s" % self.user)
        train.write_to_logger("Training samples: %d" % n_train)
        train.write_to_logger("Test samples: %d" % n_test)
        train.write_to_logger("Classes: %d" % self.n_classes)
        train.write_to_logger("Sequence length: %d" % sequence_length)
        train.write_to_logger("Step: %d" % load_data.step)
        train.write_to_logger("Shuffle: %s" % str(len(self.cv) < len(self.user_names)))
        train.write_to_logger("Factor: %d" % factor)
        train.write_to_logger("Add pitch: %s\nAdd roll: %s" % (load_data.add_pitch, load_data.add_roll))
        train.write_to_logger("Only magnitude: %s" % load_data.comp_magnitude)
        train.write_to_logger("Add filter separated signals: %s" % load_data.add_filter)
        train.write_to_logger("Differentiate: %s" % load_data.differentiate)
        train.write_to_logger("Transfer function: %s" % model.transf)
        # train.write_to_logger("Network Architecture ---------------")
        # for layer in get_all_layers(model.model):
        #    train.write_to_logger(layer.name + ": " + str(get_output_shape(layer)))

        train.train_model(f_train, train_args,
                          f_test, test_args,
                          f_validate, validate_args,
                          n_train_batches=n_train_batches,
                          n_test_batches=n_test_batches,
                          n_epochs=n_epochs)

        # Reset logging
        handlers = train.logger.handlers[:]
        for handler in handlers:
            handler.close()
            train.logger.removeHandler(handler)
        del train.logger
