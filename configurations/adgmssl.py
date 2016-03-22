import matplotlib
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from data_preparation.load_data import LoadHAR
from lasagne_extensions.nonlinearities import rectify
from models import ADGMSSL
from training.train import TrainModel
from utils.har_utils import one_hot

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()


def run_adgmssl():
    n_samples, step = 50, 25
    load_data = LoadHAR(add_pitch=False, add_roll=False, add_filter=False, n_samples=n_samples, lowpass=10, diff=False,
                        step=step, normalize='segments', comp_magnitude=True, simple_labels=False, common_labels=False)

    # datasets = [load_data.uci_hapt, load_data.idash, load_data.uci_mhealth]
    X, y, name, users, stats = load_data.uci_hapt()
    if stats is not None:
        X = np.concatenate((X, stats), axis=1)
    limited_labels = y < 18
    y = y[limited_labels]
    X = X[limited_labels]
    users = users[limited_labels]
    y_unique = np.unique(y)
    y = one_hot(y)

    # Reshape input samples to be a vector instead of samples x features
    X = np.reshape(X, (X.shape[0], -1))

    # Split into test and training
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_idx = (users != 10)  # & (users != 2) & (users != 3)
    test_idx = (users == 10)  # | (users == 2) | (users == 3)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Split training into labelled and unlabelled. Optionally stratified by the label
    X_train_labelled, X_train_unlabelled, y_train_labelled, y_train_unlabelled = \
        train_test_split(X_train, y_train, train_size=0.5, stratify=np.argmax(y_train, axis=1))

    # Add some data from test set to unlabelled training set
    X_train_unlabelled = np.concatenate((X_train_unlabelled, X_test))
    y_train_unlabelled = np.concatenate((y_train_unlabelled, y_test))

    # Combine in sets
    train_set_labelled = (X_train_labelled, y_train_labelled)
    train_set_unlabelled = (X_train_unlabelled, y_train_unlabelled)
    test_set = (X_test, y_test)
    print('Train unlabelled size: ', train_set_unlabelled[0].shape)
    print('Train labelled size: ', train_set_labelled[0].shape)
    print('Test size: ', test_set[0].shape)

    n_test = test_set[0].shape
    n, n_x = train_set_unlabelled[0].shape
    n_labelled_samples, n_x = train_set_labelled[0].shape
    n_classes = y.shape[-1]
    bs = 64
    n_batches = n//bs

    # Initialize the auxiliary deep generative model.
    model = ADGMSSL(n_x=n_x, n_a=100, n_z=100, n_y=n_classes, a_hidden=[500, 500],
                    z_hidden=[500, 500], xhat_hidden=[500, 500], y_hidden=[500, 500],
                    trans_func=rectify, batchnorm=False, x_dist='gaussian')

    # Get the training functions.
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set_unlabelled,
                                                                                          train_set_labelled,
                                                                                          test_set)
    # Update the default function arguments.
    train_args['inputs']['batchsize_unlabeled'] = bs
    train_args['inputs']['batchsize_labeled'] = n_labelled_samples
    train_args['inputs']['beta'] = 1 # 0.01 * n/n_labelled_samples
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 1  # if running a cpu: set this the no. of samples to 1.
    test_args['inputs']['samples'] = 1
    # validate_args['inputs']['samples'] = 1

    # Evaluate the approximated classification error with 100 MC samples for a good estimate.
    def error_evaluation(model, path):
        mean_evals = model.get_output(test_set[0].astype(np.float32), 100)
        t_class = np.argmax(test_set[1].astype(np.float32), axis=1)
        y_class = np.argmax(mean_evals, axis=1)
        missclass = (np.sum(y_class != t_class, dtype='float32') / len(y_class)) * 100.
        train.write_to_logger("test 100-samples misclassification: %0.2f%%." % missclass)

        plt.clf()
        fig, axarr = plt.subplots(nrows=1, ncols=2)
        axarr[0].plot(t_class, color='red')
        axarr[0].plot(y_class, linestyle='dotted')

        cm = confusion_matrix(t_class, y_class)
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        axarr[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axarr[1].set_ylabel('True')
        axarr[1].set_xlabel('Predicted')

        fig.set_size_inches(18, 10)
        fig.savefig(path, dpi=100)
        plt.close(fig)

        # Plot reconstruction
        plt.clf()
        f, axarr = plt.subplots(nrows=len(y_unique), ncols=1)
        for idx, y_l in enumerate(y_unique):
            act_idx = np.argmax(y_test, axis=1) == y_l
            test_act = test_set[0][act_idx]
            test_y = test_set[1][act_idx]

            z = model.fz(test_act, test_y, 1).eval()
            x_hat = model.f_xhat(z, test_y, 1).eval()

            axarr[idx].plot(test_act[0], color='red')
            axarr[idx].plot(x_hat[0], color='blue', linestyle='dotted')

        f.set_size_inches(12, 20)
        f.savefig(path.strip('.png') + '_fit.png', dpi=100)
        plt.close(f)

    # Define training loop. Output training evaluations every 1 epoch and the approximated good estimate
    # of the classification error every 10 epochs.
    train = TrainModel(model=model, anneal_lr=.75, anneal_lr_freq=100, output_freq=1,
                       pickle_f_custom_freq=50, f_custom_eval=error_evaluation)

    train.add_initial_training_notes("Training the auxiliary deep generative model with %i labels." % n_labelled_samples)
    train.write_to_logger("Using reduced HAPT dataset with Walking, Stairs, Inactive classes")
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
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=1000)


if __name__ == "__main__":
    run_adgmssl()
