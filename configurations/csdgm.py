from os import makedirs
from utils import copy_script, image_to_movie
from training.train import TrainModel
from lasagne_extensions.nonlinearities import rectify
from data_loaders import mnist, har
from data_loaders.data_helper import one_hot
from models.csdgm import CSDGM
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np
from data_preparation.load_data import LoadHAR
from sklearn.decomposition import PCA
import itertools
import seaborn as sns


def main():
    seed = np.random.randint(1, 2147462579)

    # def sinus_seq(period, samples, length):
    #     X = np.linspace(-np.pi*(samples/period), np.pi*(samples/period), samples)
    #     X = np.reshape(np.sin(X), (-1, length, 1))
    #     X += np.random.randn(*X.shape)*0.1
    #     X = (X - np.min(X))/(np.max(X) - np.min(X))
    #     return X, np.ones((samples/length, 1))
    #
    # X1, y1 = sinus_seq(40, 100000, 50)
    # X2, y2 = sinus_seq(20, 40000, 50)
    #
    # X = np.concatenate((X1, X2)).astype('float32')
    # y = np.concatenate((y1*0, y2*1), axis=0).astype('int')
    #
    # dim_samples, dim_sequence, dim_features = X.shape
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # X, y, users, stats = har.load()

    n_samples, step = 50, 50
    load_data = LoadHAR(add_pitch=False, add_roll=False, add_filter=False, n_samples=n_samples, diff=False,
                        step=step, normalize='segments', comp_magnitude=True, simple_labels=False, common_labels=False)
    X, y, name, users, stats = load_data.uci_hapt()

    limited_labels = y < 3
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
    num_classes = len(y_unique)

    # Split into train and test stratified by users
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500, stratify=users)

    n_samples = 1000
    # Split training into labelled and unlabelled. Optionally stratified by the label
    X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = \
        train_test_split(X_train, y_train, train_size=n_samples, stratify=np.argmax(y_train, axis=1))

    # Combine in sets
    train_set_labeled = (X_train_labeled, y_train_labeled)
    train_set_unlabeled = (X_train_unlabeled, y_train_unlabeled)
    test_set = (X_test, y_test)
    print('Train unlabelled size: ', train_set_unlabeled[0].shape)
    print('Train labelled size: ', train_set_labeled[0].shape)
    print('Test size: ', test_set[0].shape)

    n, n_l, n_c = train_set_unlabeled[0].shape  # Datapoints in the dataset, input features.
    n_batches = n / 100  # The number of batches.
    bs = n / n_batches  # The batchsize.

    # Initialize the auxiliary deep generative model.
    # [num_filters, stride, pool]
    filters = [[128, 1, 2],
               [128, 1, 2],
               [128, 1, 2],
               [128, 1, 2]]
    model = CSDGM(n_c=int(n_c), n_l=int(n_l), n_a=100, n_z=128, n_y=num_classes, qa_hid=[100],
                  qz_hid=[100], qy_hid=[100], px_hid=[128], pa_hid=[100], filters=filters,
                  nonlinearity=rectify, batchnorm=False, x_dist='gaussian')

    # Copy script to output folder
    copy_script(__file__, model)

    # Create output path for PCA plot
    makedirs(model.get_root_path() + '/training custom evals/pca')

    # Get the training functions.
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set_unlabeled, train_set_labeled, test_set)
    # Update the default function arguments.
    train_args['inputs']['batchsize_unlabeled'] = bs
    train_args['inputs']['batchsize_labeled'] = n_samples
    train_args['inputs']['beta'] = .1
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 1
    train_args['inputs']['warmup'] = 1.1

    def custom_evaluation(model, path):
        # Get model output
        x_ = test_set[0]
        y_ = test_set[1]

        # qy = model.f_qy(x_, 1)
        qa = model.f_qa(x_, 1)
        qz = model.f_qz(x_, y_, 1)
        # pa = model.f_pa(qz, y_, 1)
        px = model.f_px(x_, qa, qz, y_, 1)
        px_mu = model.f_mu(x_, qa, qz, y_, 1)
        px_var = np.exp(model.f_var(x_, qa, qz, y_, 1))

        # reduce y to integers
        y_ = np.argmax(y_, axis=1)

        plt.clf()
        f, axarr = plt.subplots(nrows=len(y_unique), ncols=2)
        for idx, y_l in enumerate(y_unique):
            l_idx = y_ == y_l

            axarr[idx, 0].plot(x_[l_idx][:2].reshape(-1, n_c))
            axarr[idx, 0].plot(px[l_idx][:2].reshape(-1, n_c), linestyle='dotted')
            axarr[idx, 1].plot(px_mu[l_idx][:2].reshape(-1, n_c), label="mu")
            axarr[idx, 1].plot(px_var[l_idx][:2].reshape(-1, n_c), label="var")
            plt.legend()

        f.set_size_inches(12, 8)
        f.savefig(path, dpi=100, format='png')
        plt.close(f)

        # Plot PCA decomp
        z_pca = PCA(n_components=2).fit_transform(qz)
        a_pca = PCA(n_components=2).fit_transform(qa)

        palette = itertools.cycle(sns.color_palette())
        plt.clf()
        plt.figure()
        f, axarr = plt.subplots(ncols=2)
        for i in set(y_unique):
            c = next(palette)
            axarr[0].scatter(z_pca[y_ == i, 0], z_pca[y_ == i, 1], c=c, alpha=0.8)
            axarr[1].scatter(a_pca[y_ == i, 0], a_pca[y_ == i, 1], c=c, alpha=0.8, label=str(i))
        plt.legend()
        axarr[0].set_title('Z')
        axarr[1].set_title('A')
        f.set_size_inches(10, 6)
        plt.savefig(path.replace('custom_eval_plot', 'pca/z'),  dpi=100, format='png')
        plt.close()

    # Define training loop. Output training evaluations every 1 epoch
    # and the custom evaluation method every 10 epochs.
    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=custom_evaluation)
    train.add_initial_training_notes("Training the rae with bn %s. seed %i." % (str(model.batchnorm), seed))
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=1000,
                      anneal=[("learningrate", 100, 0.75, 3e-5),
                              ("warmup", 5, 0.99, 0.1)])

    # image_to_movie.create(model.get_root_path() + '/training_custom_evals/', rate=3)

if __name__ == "__main__":
    main()
