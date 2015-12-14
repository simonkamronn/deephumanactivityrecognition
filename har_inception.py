from models.inception_sequence import Inception_seq
from training.train import TrainModel
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
from lasagne.layers import get_all_layers, get_output_shape
import load_data as ld


def main():
    add_pitch, add_roll, add_filter = True, True, True
    n_samples, step = 200, 50
    shuffle = True
    batch_size = 64
    (train_set, test_set, valid_set, (sequence_length, n_features, n_classes)), name = \
        ld.LoadHAR().uci_hapt(add_pitch=add_pitch, add_roll=add_roll, add_filter=add_filter,
                              n_samples=n_samples, step=step, shuffle=shuffle)
    n_train = train_set[0].shape[0]
    n_test = test_set[0].shape[0]

    n_train_batches = n_train//batch_size
    n_test_batches = n_test//batch_size
    n_valid_batches = n_test//batch_size

    print("n_train_batches: %d, n_test_batches: %d" % (n_train_batches, n_test_batches))
    # num_1x1, num_2x1_proj, reduce_3x1, num_3x1, reduce_5x1, num_5x1
    model = Inception_seq(n_in=(sequence_length, n_features),
                          inception_layers=[(32, 16, 32, 64, 8, 16),
                                            (32, 16, 32, 64, 8, 16)],
                          pool_sizes=[2, 2],
                          n_hidden=[0],
                          dropout_probability=0.5,
                          n_out=n_classes,
                          trans_func=rectify,
                          out_func=softmax,
                          batch_size=batch_size,
                          batch_norm=False)

    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set,
                                                                                          test_set,
                                                                                          valid_set)
    train_args['inputs']['batchsize'] = batch_size
    train_args['inputs']['learningrate'] = 0.003
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 1e-6

    test_args['inputs']['batchsize'] = batch_size
    validate_args['inputs']['batchsize'] = batch_size

    model.log += "\nDataset: %s" % name
    model.log += "\nTraining samples: %d" % n_train
    model.log += "\nTest samples: %d" % n_test
    model.log += "\nSequence length: %d" % sequence_length
    model.log += "\nStep: %d" % step
    model.log += "\nShuffle %s" % shuffle
    model.log += "\nAdd pitch: %s\nAdd roll: %s" % (add_pitch, add_roll)
    model.log += "\nAdd filter separated signals: %s" % add_filter
    model.log += "\nTransfer function: %s" % model.transf
    model.log += "\nNetwork Architecture ---------------"
    for layer in get_all_layers(model.model):
        # print(layer.name, ": ", get_output_shape(layer))
        model.log += "\n" + layer.name + ": " + str(get_output_shape(layer))

    train = TrainModel(model=model,
                       anneal_lr=0.90,
                       anneal_lr_freq=50,
                       output_freq=1,
                       pickle_f_custom_freq=100,
                       f_custom_eval=None)
    train.pickle = True
    train.add_initial_training_notes("")
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_train_batches,
                      n_test_batches=n_test_batches,
                      n_valid_batches=n_valid_batches,
                      n_epochs=2000)

if __name__ == "__main__":
    main()
