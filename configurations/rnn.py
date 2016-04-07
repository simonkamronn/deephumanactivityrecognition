from os import rmdir
from lasagne.nonlinearities import rectify, softmax
from base import ModelConfiguration
from data_preparation.load_data import LoadHAR
from models.rnn import RNN
from training.train import TrainModel
from utils import env_paths as paths


def main():
    add_pitch, add_roll, add_filter = False, False, True
    n_samples, step = 50, 50
    load_data = LoadHAR(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                        n_samples=n_samples, step=step)

    conf = ModelConfiguration()
    conf.load_datasets([load_data.uci_mhealth, load_data.idash, load_data.wisdm1])

    user_idx = -1
    user = None
    # Create a time-string for our cv run
    if user is not None:
        train_idx = conf.users != user
        test_idx = conf.users == user
        conf.cv = ((train_idx, test_idx), )

    for train_index, test_index in conf.cv:
        conf.user = user

        model = RNN(n_in=(n_samples, conf.n_features),
                    n_hidden=[50, 50],
                    dropout_probability=0.5,
                    n_out=conf.n_classes,
                    ccf=False,
                    trans_func=rectify,
                    out_func=softmax)

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
                           anneal_lr_freq=50,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=None)
        train.pickle = False

        conf.run(train_index, test_index, lr=0.002, n_epochs=300, model=model, train=train, load_data=load_data)

if __name__ == "__main__":
    main()
