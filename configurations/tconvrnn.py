from os import rmdir, path
import shutil
from lasagne.nonlinearities import leaky_rectify, rectify, softmax
from base import ModelConfiguration
from data_preparation.load_data import LoadHAR
from models.tconvrnn import tconvRNN
from training.train import TrainModel
from utils import env_paths as paths
from sklearn.cross_validation import LeavePLabelOut, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
import numpy as np


def main():
    n_samples, step = 40, 20
    load_data = LoadHAR(add_pitch=False, add_roll=False, add_filter=False, n_samples=n_samples,
                        step=step, normalize='channels', comp_magnitude=False, simple_labels=True, common_labels=False)

    conf = ModelConfiguration()
    conf.load_datasets([load_data.uci_hapt], label_limit=6)

    user_idx = -1
    user = None  # 'UCI HAPT10'
    if user is not None:
        train_idx = conf.users != user
        test_idx = conf.users == user
        conf.cv = ((train_idx, test_idx), )
        print('Testing user: %s' % user)
    else:
        # Cross validate on users
        conf.cv = LeavePLabelOut(conf.users, p=1)

        # Divide into K folds balanced on labels
        # conf.cv = StratifiedKFold(conf.users, n_folds=10)

        # And shuffle
        # conf.cv = StratifiedShuffleSplit(np.argmax(conf.y, axis=1), n_iter=1, test_size=0.1, random_state=None)

        # Pure shuffle
        # conf.cv = ShuffleSplit(conf.y.shape[0], n_iter=2, test_size=0.1)

    for train_index, test_index in conf.cv:
        conf.user = user

        model = tconvRNN(n_in=(n_samples, conf.n_features),
                         n_filters=[64, 64, 64, 64],
                         filter_sizes=[5]*4,
                         pool_sizes=[0]*4,
                         n_hidden=[128, 128],
                         conv_dropout=0.3,
                         rnn_in_dropout=0.0,
                         rnn_hid_dropout=0.0,
                         output_dropout=0.5,
                         n_out=conf.n_classes,
                         trans_func=leaky_rectify,
                         out_func=softmax,
                         stats=conf.stats)

        if len(conf.cv) > 1:
            user_idx += 1
            if len(conf.cv) == len(conf.user_names):
                conf.user = conf.user_names[user_idx]
            else:
                conf.user = conf.name + ' K_%d' % user_idx

            # Generate root path and edit
            root_path = model.get_root_path()
            model.root_path = "%s_cv_%s_%s" % (root_path, conf.d, conf.user)
            paths.path_exists(model.root_path)
            rmdir(root_path)

        scriptpath = path.realpath(__file__)
        filename = path.basename(scriptpath)
        print(scriptpath, model.root_path, filename)
        shutil.copy(scriptpath, model.root_path + '/' + filename)

        train = TrainModel(model=model,
                           anneal_lr=0.9,
                           anneal_lr_freq=1,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=None)
        train.pickle = False
        train.write_to_logger("Using StratifiedShuffleSplit with n_iter=1, test_size=0.1, random_state=None")

        conf.run(train_index,
                 test_index,
                 lr=0.003,
                 n_epochs=500,
                 model=model,
                 train=train,
                 load_data=load_data,
                 batch_size=100)

if __name__ == "__main__":
    main()
