from os import rmdir, path
import shutil
from lasagne.nonlinearities import rectify
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

def main():
    n_samples, step = 100, 100
    load_data = LoadHAR(add_pitch=True, add_roll=True, add_filter=False, n_samples=n_samples, lowpass=10, diff=True,
                        step=step, normalize='segments', comp_magnitude=True, simple_labels=False, common_labels=False)

    X, y, name, users, stats = load_data.uci_hapt()
    users = ['%s%02d' % (name, user) for user in users]
    limited_labels = y < 20
    y = y[limited_labels]
    X = X[limited_labels].astype('float32')
    users = np.char.asarray(users)[limited_labels]
    y_unique = np.unique(y)

    cv = StratifiedShuffleSplit(y, n_iter=1, test_size=0.1, random_state=0)
    for (train_index, test_index) in cv:
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    n_win, n_samples, n_features = x_train.shape

    train_set = (x_train, y_train)
    test_set = (x_test, y_test)
    print('Train size: ', train_set[0].shape)
    print('Test size: ', test_set[0].shape)

    n_train = train_set[0].shape[0]
    n_test = test_set[0].shape[0]
    batch_size = 64

    n_test_batches = n_test//batch_size
    n_train_batches = n_train//batch_size

    model = CAE(n_in=(int(n_samples), int(n_features)),
                filters=[128, 64, 32],
                n_hidden=32,
                n_out=n_samples,
                trans_func=rectify,
                stats=0)

    # Build model
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                          test_set,
                                                                                          None)

    def f_custom(model, path):
        plt.clf()
        f, axarr = plt.subplots(nrows=len(y_unique), ncols=1)

        for idx, y_l in enumerate(y_unique):
            act_idx = y_test == y_l
            test_act = test_set[0][act_idx]
            out = model.get_output(test_act).eval()

            axarr[idx].plot(test_act[0], color='red')
            axarr[idx].plot(out[0], color='blue', linestyle='dotted')

        f.set_size_inches(12, 20)
        f.savefig(path, dpi=100)
        plt.close(f)

    train = TrainModel(model=model,
                       anneal_lr=0.75,
                       anneal_lr_freq=100,
                       output_freq=1,
                       pickle_f_custom_freq=100,
                       f_custom_eval=f_custom)
    train.pickle = False

    train.write_to_logger("Normalizing: %s" % load_data.normalize)
    train.write_to_logger("Simple labels: %s" % load_data.simple_labels)
    train.write_to_logger("Common labels: %s" % load_data.common_labels)
    train.write_to_logger("Step: %d" % load_data.step)
    train.write_to_logger("Add pitch: %s\nAdd roll: %s" % (load_data.add_pitch, load_data.add_roll))
    train.write_to_logger("Only magnitude: %s" % load_data.comp_magnitude)
    train.write_to_logger("Lowpass: %s" % str(load_data.lowpass))
    train.write_to_logger("Add filter separated signals: %s" % load_data.add_filter)

    test_args['inputs']['batchsize'] = batch_size
    train_args['inputs']['batchsize'] = batch_size
    train_args['inputs']['learningrate'] = 0.003
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 1e-6
    validate_args['inputs']['batchsize'] = batch_size

    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_train_batches,
                      n_test_batches=n_test_batches,
                      n_epochs=2000)

if __name__ == "__main__":
    main()
