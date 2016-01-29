from os import rmdir
from lasagne.nonlinearities import rectify, softmax
from base import ModelConfiguration
from data_preparation.load_data import LoadHAR
from models.cnn import CNN
from training.train import TrainModel
from utils import env_paths as paths
from sklearn.cross_validation import LeaveOneLabelOut


def main():
    n_samples, step = 100, 50
    load_data = LoadHAR(add_pitch=False, add_roll=False, add_filter=True,
                        n_samples=n_samples, step=step, normalize=True)

    conf = ModelConfiguration()
    conf.load_datasets([load_data.uci_hapt], label_limit=18)  # , load_data.uci_mhealth, load_data.idash

    user_idx = -1
    user = None
    # Create a time-string for our cv run
    if user is not None:
        train_idx = conf.users != user
        test_idx = conf.users == user
        conf.cv = ((train_idx, test_idx), )
    else:
        conf.cv = LeaveOneLabelOut(conf.users)

    for train_index, test_index in conf.cv:
        conf.user = user

        model = CNN(n_in=(n_samples+2, conf.n_features),
                    n_filters=[64, 64, 64, 64],
                    filter_sizes=[5, 5, 3, 3],
                    pool_sizes=[2, 2, 2, 2],
                    conv_dropout=0.5,
                    n_hidden=[128],
                    dense_dropout=0.5,
                    n_out=conf.n_classes,
                    ccf=False,
                    trans_func=rectify,
                    out_func=softmax,
                    batch_norm=True,
                    input_noise=0.2,
                    stats=2)

        if len(conf.cv) > 1:
            user_idx += 1
            conf.user = conf.user_names[user_idx]

            # Generate root path and edit
            root_path = model.get_root_path()
            model.root_path = "%s_cv_%s_%s" % (root_path, conf.d, conf.user)
            paths.path_exists(model.root_path)
            rmdir(root_path)

        train = TrainModel(model=model,
                           anneal_lr=0.75,
                           anneal_lr_freq=100,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=None)
        train.pickle = False

        conf.run(train_index, test_index, lr=0.003, n_epochs=300, model=model, train=train, load_data=load_data)

if __name__ == "__main__":
    main()
