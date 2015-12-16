from models.rcnn import RCNN
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
from lasagne.layers import get_all_layers, get_output_shape
import load_data as ld
from sklearn.cross_validation import LeaveOneLabelOut
import numpy as np


def main():
    add_pitch, add_roll, add_filter = False, False, True
    n_samples, step = 200, 100
    shuffle = False
    batch_size = 32
    (train_set, test_set, valid_set, (sequence_length, n_features, n_classes)), name, users = \
        ld.LoadHAR().uci_hapt(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                              n_samples=n_samples, step=step, shuffle=shuffle)

    X = np.concatenate((train_set[0], test_set[0]), axis=0)
    y = np.concatenate((train_set[1], test_set[1]), axis=0)

    n_conv = 2
    model = RCNN(n_in=(sequence_length, n_features),
                 n_filters=[64]*n_conv,
                 filter_sizes=[3]*n_conv,
                 pool_sizes=[2]*n_conv,
                 rcl=[3, 3, 3],
                 rcl_dropout=0.5,
                 n_hidden=[512],
                 dropout_probability=0.5,
                 n_out=n_classes,
                 downsample=1,
                 ccf=False,
                 trans_func=rectify,
                 out_func=softmax,
                 batch_size=batch_size,
                 batch_norm=False)
    base_params = model.model_params

    lol = LeaveOneLabelOut(users)
    user = 0
    eval_validation = []
    for train_index, test_index in lol:
        user += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_set = (X_train, y_train)
        test_set = (X_test, y_test)

        n_train = train_set[0].shape[0]
        n_test = test_set[0].shape[0]

        n_train_batches = n_train//batch_size
        n_test_batches = n_test//batch_size
        n_valid_batches = n_test//batch_size

        print('Train set shape: ', X_train.shape)
        print('Test set shape: ', X_test.shape)
        print("n_train_batches: %d, n_test_batches: %d" % (n_train_batches, n_test_batches))

        model.model_params = base_params
        f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                              test_set,
                                                                                              test_set)
        test_args['inputs']['batchsize'] = batch_size
        validate_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['batchsize'] = batch_size
        train_args['inputs']['learningrate'] = 0.002
        train_args['inputs']['beta1'] = 0.9
        train_args['inputs']['beta2'] = 1e-6

        train = TrainModel(model=model,
                           anneal_lr=0.90,
                           anneal_lr_freq=50,
                           output_freq=1,
                           pickle_f_custom_freq=100,
                           f_custom_eval=None)
        train.pickle = True

        train.add_initial_training_notes("")
        train.write_to_logger("Dataset: %s" % name)
        train.write_to_logger("LOO user: %d" % user)
        train.write_to_logger("Training samples: %d" % n_train)
        train.write_to_logger("Test samples: %d" % n_test)
        train.write_to_logger("Sequence length: %d" % sequence_length)
        train.write_to_logger("Step: %d" % step)
        train.write_to_logger("Shuffle: %s" % shuffle)
        train.write_to_logger("Add pitch: %s\nAdd roll: %s" % (add_pitch, add_roll))
        train.write_to_logger("Add filter separated signals: %s" % add_filter)
        train.write_to_logger("Transfer function: %s" % model.transf)
        train.write_to_logger("Network Architecture ---------------")
        for layer in get_all_layers(model.model):
            # print(layer.name, ": ", get_output_shape(layer))
            train.write_to_logger(layer.name + ": " + str(get_output_shape(layer)))

        train.train_model(f_train, train_args,
                          f_test, test_args,
                          f_validate, validate_args,
                          n_train_batches=n_train_batches,
                          n_test_batches=n_test_batches,
                          n_valid_batches=n_valid_batches,
                          n_epochs=100)

        # Collect
        eval_validation.append(np.max(train.eval_validation, axis=0))
        print(eval_validation)

        # Reset logging
        handlers = train.logger.handlers[:]
        for handler in handlers:
            handler.close()
            train.logger.removeHandler(handler)
        del train.logger


if __name__ == "__main__":
    main()
