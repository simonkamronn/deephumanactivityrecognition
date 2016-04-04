from os import makedirs
from utils import copy_script, image_to_movie
from training.train import TrainModel
from lasagne_extensions.nonlinearities import rectify, softplus
from data_loaders import mnist, har
from data_loaders.data_helper import one_hot
from models.rvae import RVAE
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cross_validation import train_test_split
from data_preparation.load_data import LoadHAR
from sklearn.decomposition import PCA
import itertools
import seaborn as sns


def run_vrae_har():
    seed = np.random.randint(1, 2147462579)

    # def sinus_seq(period, samples, length):
    #     X = np.linspace(-np.pi*(samples/period), np.pi*(samples/period), samples)
    #     X = np.reshape(np.sin(X), (-1, length, 1))
    #     X += np.random.randn(*X.shape)*0.1
    #     # X = (X - np.min(X))/(np.max(X) - np.min(X))
    #     return X, np.ones((samples/length, 1))
    #
    # X1, y1 = sinus_seq(20, 100000, 40)
    # X2, y2 = sinus_seq(12, 100000, 40)
    # X3, y3 = sinus_seq(8, 100000, 40)
    #
    # X = np.concatenate((X1, X2, X3)).astype('float32')
    # y = np.concatenate((y1*0, y2*1, y3*2), axis=0).astype('int')[:, 0]
    # y_unique = np.unique(list(y))
    #
    # y = one_hot(y, len(y_unique))
    #
    # dim_samples, dim_sequence, dim_features = X.shape
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    ##
    # HAR data
    # X, y, users, stats = har.load()

    n_samples, step = 50, 25
    load_data = LoadHAR(add_pitch=False, add_roll=False, add_filter=False, n_samples=n_samples, diff=False,
                        step=step, normalize='segments', comp_magnitude=False, simple_labels=False, common_labels=False)
    X, y, name, users, stats = load_data.uci_hapt()

    limited_labels = y < 18
    y = y[limited_labels]
    X = X[limited_labels].astype(np.float32)
    users = users[limited_labels]

    # X -= X.mean(axis=0)

    # Compress labels
    for idx, label in enumerate(np.unique(y)):
        if not np.equal(idx, label):
            y[y == label] = idx

    y_unique = np.unique(y)
    y = one_hot(y, len(y_unique))

    dim_samples, dim_sequence, n_c = X.shape
    num_classes = len(y_unique)

    # Split into train and test stratified by users
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=1)
    ##

    # Combine in sets
    train_set = (X_train, y_train)
    test_set = (X_test, y_test)
    print('Train size: ', train_set[0].shape)
    print('Test size: ', test_set[0].shape)

    n, n_l, n_c = train_set[0].shape  # Datapoints in the dataset, input features.
    n_batches = n / 100  # The number of batches.
    bs = n / n_batches  # The batchsize.

    # Initialize the auxiliary deep generative model.
    model = RVAE(n_c=n_c, n_z=128, qz_hid=[256, 256], px_hid=[256, 256], enc_rnn=256, dec_rnn=256, n_l=n_l,
                 nonlinearity=rectify, batchnorm=False, x_dist='gaussian', px_nonlinearity=None)

    # Copy script to output folder
    copy_script(__file__, model)

    # Create output path for PCA plot
    makedirs(model.get_root_path() + '/training custom evals/pca')

    # Get the training functions.
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set, test_set)
    # Update the default function arguments.
    train_args['inputs']['batchsize'] = bs
    train_args['inputs']['learningrate'] = 1e-3
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 1
    train_args['inputs']['warmup'] = 1.1

    def custom_evaluation(model, path):
        # Get model output
        x_ = test_set[0]
        y_ = test_set[1]

        qz = model.f_qz(x_, 1)
        px = model.f_px(x_, qz, 1)
        px_mu = model.f_mu(x_, qz, 1)
        px_var = np.exp(model.f_var(x_, qz, 1))

        # reduce y to integers
        y_ = np.argmax(y_, axis=1)

        plt.clf()
        f, axarr = plt.subplots(nrows=num_classes, ncols=n_c*2)
        for idx, y_l in enumerate(y_unique):
            l_idx = y_ == y_l

            for c in range(n_c):
                axarr[idx, c*2].plot(x_[l_idx, :, c][:2].reshape(-1))
                axarr[idx, c*2].plot(px[l_idx, :, c][:2].reshape(-1), linestyle='dotted')
                axarr[idx, c*2 + 1].plot(px_mu[l_idx, :, c][:2].reshape(-1), label="mu")
                axarr[idx, c*2 + 1].plot(px_var[l_idx, :, c][:2].reshape(-1), label="var")
            plt.legend()

        f.set_size_inches(20, num_classes*3)
        f.savefig(path, dpi=100, format='png')
        plt.close(f)

        # Plot PCA decomp
        z_pca = PCA(n_components=2).fit_transform(qz)

        palette = itertools.cycle(sns.color_palette())
        plt.clf()
        plt.figure()
        for i in set(y_unique):
            plt.scatter(z_pca[y_ == i, 0], z_pca[y_ == i, 1], c=next(palette), alpha=0.8)
        plt.legend()
        plt.title('PCA of Z')
        plt.savefig(path.replace('custom_eval_plot', 'pca/z'))
        plt.close()

    def anneal_func(input):
        return input - 0.01

    # Define training loop. Output training evaluations every 1 epoch
    # and the custom evaluation method every 10 epochs.
    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=custom_evaluation)
    train.add_initial_training_notes("Training the vrae with bn %s. seed %i." % (str(model.batchnorm), seed))
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=1000,
                      # Any symbolic model variable can be annealed during
                      # training with a tuple of (var_name, every, scale constant, minimum value).
                      anneal=[("learningrate", 100, 0.75, 3e-5),
                              ("warmup", 1, anneal_func, 0.1)])

    image_to_movie.create(model.get_root_path() + '/training custom evals', rate=3)


if __name__ == "__main__":
    run_vrae_har()
