from os import rmdir, path
import shutil
from lasagne.nonlinearities import rectify, leaky_rectify
from data_preparation.load_data import LoadHAR
from models.cae import CAE
from training.train import TrainModel
from utils import env_paths as paths
from sklearn.cross_validation import LeaveOneLabelOut, StratifiedShuffleSplit
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
from data_loaders.data_helper import one_hot
from sklearn.cross_validation import train_test_split
from utils import copy_script, image_to_movie


def main():
    n_samples, step = 50, 25
    load_data = LoadHAR(add_pitch=False, add_roll=False, add_filter=False, n_samples=n_samples, diff=False,
                        step=step, normalize='segments', comp_magnitude=False, simple_labels=False, common_labels=False)
    X, y, name, users, stats = load_data.uci_hapt()

    limited_labels = y < 18
    y = y[limited_labels]
    X = X[limited_labels].astype(np.float32)
    users = users[limited_labels]

    # Compress labels
    for idx, label in enumerate(np.unique(y)):
        if not np.equal(idx, label):
            y[y == label] = idx

    y_unique = np.unique(y)
    y = one_hot(y, len(y_unique))
    num_classes = len(y_unique)

    # Split into train and test stratified by users
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=np.argmax(y, axis=1))

    # Combine in sets
    train_set = (X_train, y_train)
    test_set = (X_test, y_test)
    print('Train size: ', train_set[0].shape)
    print('Test size: ', test_set[0].shape)

    n, n_l, n_c = train_set[0].shape  # Datapoints in the dataset, input features.
    n_batches = n / 100  # The number of batches.
    bs = n / n_batches  # The batchsize.

    model = CAE(n_in=(int(n_l), int(n_c)),
                filters=[8, 16, 32, 64],
                n_hidden=64,
                n_out=n_samples,
                trans_func=leaky_rectify,
                stats=0)

    # Copy script to output folder
    copy_script(__file__, model)

    # Build model
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                          test_set,
                                                                                          None)

    def custom_evaluation(model, path):
        # Get model output
        x_ = test_set[0]
        y_ = test_set[1]
        xhat = model.f_px(x_)

        # reduce y to integers
        y_ = np.argmax(y_, axis=1)

        plt.clf()
        f, axarr = plt.subplots(nrows=num_classes, ncols=n_c)
        for idx, y_l in enumerate(y_unique):
            l_idx = y_ == y_l

            for c in range(n_c):
                axarr[idx, c].plot(x_[l_idx, :, c][:2].reshape(-1), color='red')
                axarr[idx, c].plot(xhat[l_idx, :, c][:2].reshape(-1), color='blue', linestyle='dotted')

        f.set_size_inches(12, 3*num_classes)
        f.savefig(path, dpi=100, format='png')
        plt.close(f)

    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=custom_evaluation)
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=10000,
                      anneal=[("learningrate", 100, 0.75, 3e-5)])
    train.pickle = False

    train.write_to_logger("Normalizing: %s" % load_data.normalize)
    train.write_to_logger("Simple labels: %s" % load_data.simple_labels)
    train.write_to_logger("Common labels: %s" % load_data.common_labels)
    train.write_to_logger("Sequence length: %d" % load_data.n_samples)
    train.write_to_logger("Step: %d" % load_data.step)
    train.write_to_logger("Add pitch: %s\nAdd roll: %s" % (load_data.add_pitch, load_data.add_roll))
    train.write_to_logger("Only magnitude: %s" % load_data.comp_magnitude)
    train.write_to_logger("Lowpass: %s" % str(load_data.lowpass))
    train.write_to_logger("Add filter separated signals: %s" % load_data.add_filter)
    train.write_to_logger("Differentiate: %s" % load_data.differentiate)

    train_args['inputs']['batchsize'] = 100
    train_args['inputs']['learningrate'] = 1e-3
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999

    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=2000,
                      anneal=[("learningrate", 100, 0.75, 3e-5)])

if __name__ == "__main__":
    main()
