import theano
from training.train import TrainModel
from lasagne_extensions.nonlinearities import rectify
from data_loaders import mnist, har
from data_loaders.data_helper import one_hot
from models.rae import RAE
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np


def run_rae_har():
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

    X, y, users, stats = har.load()

    limited_labels = y < 5
    y = y[limited_labels]
    X = X[limited_labels]
    users = users[limited_labels]

    # Compress labels
    for idx, label in enumerate(np.unique(y)):
        if not np.equal(idx, label):
            y[y == label] = idx

    y_unique = np.unique(y)
    y = one_hot(y, len(y_unique))

    dim_samples, dim_sequence, dim_features = X.shape
    num_classes = len(y_unique)

    # Split into train and test stratified by users
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=users)

    # Combine in sets
    train_set = (X_train, y_train)
    test_set = (X_test, y_test)
    print('Train size: ', train_set[0].shape)
    print('Test size: ', test_set[0].shape)

    n, seq, n_x = train_set[0].shape  # Datapoints in the dataset, input features.
    n_batches = n / 100  # The number of batches.
    bs = n / n_batches  # The batchsize.

    # Initialize the auxiliary deep generative model.
    model = RAE(n_x=int(n_x), px_hid=[128], enc_rnn=128, dec_rnn=128, seq_length=int(seq),
                nonlinearity=rectify, batchnorm=False)

    # Get the training functions.
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set, test_set)
    # Update the default function arguments.
    train_args['inputs']['batchsize'] = 100
    train_args['inputs']['learningrate'] = 1e-3
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999

    def custom_evaluation(model, path):
        plt.clf()
        f, axarr = plt.subplots(nrows=len(np.unique(y)), ncols=1)
        for idx, y_l in enumerate(np.unique(y)):
            act_idx = test_set[1] == y_l
            test_act = test_set[0][act_idx[:, 0]]

            xhat = model.f_px(test_act)

            axarr[idx].plot(test_act[:3].reshape(-1, dim_features), color='red')
            axarr[idx].plot(xhat[:3].reshape(-1, dim_features), color='blue', linestyle='dotted')

        f.set_size_inches(8, 5)
        f.savefig(path, dpi=100, format='png')
        plt.close(f)

    # Define training loop. Output training evaluations every 1 epoch
    # and the custom evaluation method every 10 epochs.
    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=custom_evaluation)
    train.add_initial_training_notes("Training the rae with bn %s. seed %i." % (str(model.batchnorm), seed))
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=10000,
                      anneal=[("learningrate", 100, 0.75, 3e-5)])

if __name__ == "__main__":
    run_rae_har()
