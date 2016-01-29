from os import rmdir
from lasagne.nonlinearities import rectify, softmax
from base import ModelConfiguration
from data_preparation.load_data import LoadHAR
from models.rcnn import RCNN
from training.train import TrainModel
from utils import env_paths as paths
from sklearn.cross_validation import LeaveOneLabelOut


def main():
    add_pitch, add_roll, add_filter = False, False, True
    n_samples, step = 200, 200
    load_data = LoadHAR(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                        n_samples=n_samples, step=step, normalize=True)

    conf = ModelConfiguration()
    conf.load_datasets([load_data.uci_hapt, load_data.uci_mhealth, load_data.idash, load_data.wisdm1])

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

        n_conv = 1
        model = RCNN(n_in=(n_samples, conf.n_features),
                     n_filters=[32, 64, 128, 256],
                     filter_sizes=[3]*n_conv,
                     pool_sizes=[2]*n_conv,
                     rcl=[2, 2, 2, 2],
                     rcl_dropout=0.5,
                     n_hidden=[512],
                     dropout_probability=0.5,
                     n_out=conf.n_classes,
                     downsample=1,
                     ccf=False,
                     trans_func=rectify,
                     out_func=softmax,
                     batch_norm=True)

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

        conf.run(train_index, test_index, lr=0.003, n_epochs=200, model=model, train=train, load_data=load_data)

if __name__ == "__main__":
    main()
