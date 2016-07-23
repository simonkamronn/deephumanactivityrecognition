import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import normalize, MinMaxScaler
from lasagne.nonlinearities import leaky_rectify, softmax, rectify
from training.train import TrainModel
from utils import copy_script

from models.sphere_window_convrnn import wconvRNN
from models.sphere_rnn_brnn import HRNN
from models.sphere_brnn import BRNN
from models.sphere_cnn import CNN

import pendulum
import os


def load_sequence(file_id, test_train, data_path=""):
    filename = str(file_id).zfill(5)

    df = pd.read_csv(data_path + '/{}/{}/columns_20.csv'.format(test_train, filename))
    data = df.values

    if test_train == 'train':
        target = pd.read_csv(data_path + '/{}/{}/targets.csv'.format(test_train, filename)).fillna(0)
        target = target.values[:, 2:]
    else:
        target = []

    return data, target


def load_sequences(file_ids, data_path, test_train='train', fs=10):
    x_es = []
    y_es = []

    for file_id in file_ids:
        data, target = load_sequence(file_id, test_train, data_path)
        data = data[:target.shape[0] * fs]

        x_es.append(data)
        y_es.append(target)

    return np.row_stack(x_es), np.row_stack(y_es)

# Set parameters
data_path = '/nobackup/titans/sdka/data/sphere/data'
class_weights = np.asarray(json.load(open(data_path+'/class_weights.json', 'r')))
train_ids = range(1, 9)
test_ids = range(9, 11)
fs = 20
num_secs = 29
batch_size = 25

# Load data
train_x, train_y = load_sequences(train_ids, data_path, fs=fs)
test_x, test_y = load_sequences(test_ids, data_path, fs=fs)

# Scale columns
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_x)

train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

# Calc max seconds
train_data_lim = (train_x.shape[0] // (fs * num_secs)) * (fs * num_secs)
test_data_lim = (test_x.shape[0] // (fs * num_secs)) * (fs * num_secs)

# Reshape to 29 second windows
train_x = train_x[:train_data_lim].reshape(-1, fs * num_secs, 29).astype('float32')
test_x = test_x[:test_data_lim].reshape(-1, fs * num_secs, 29).astype('float32')

train_y = train_y[:int(train_data_lim/fs)].reshape(-1, num_secs, 20)
test_y = test_y[:int(test_data_lim/fs)].reshape(-1, num_secs, 20)

# Fill up to match batch_size
filling = np.zeros((3, fs * num_secs, 29))
filling[:, :, [9, 13, 18, 28]] = 1
train_x = np.concatenate((train_x, filling))
train_y = np.pad(train_y, ((0, 3), (0, 0), (0, 0)), mode='constant')

print("Training sizes")
print(train_x.shape)
print(train_y.shape)

print("Testing sizes")
print(test_x.shape)
print(test_y.shape)

# Set parameters
n_samples, sequence_length, n_features = train_x.shape
n_classes = train_y.shape[-1]

train_set = (train_x, train_y)
test_set = (test_x, test_y)
n_train_batches = train_x.shape[0] // batch_size
n_test_batches = test_x.shape[0] // batch_size

# Slicers for cutting out columns
slicers = dict(pir=slice(0, 10), accel=slice(10, 14), rssi=slice(14, 19), video=slice(19, None))

# model = CNN(n_in=(sequence_length, n_features),
#             n_filters=[32, 64],
#             filter_sizes=[3, 3],
#             n_hidden=[256],
#             n_out=n_classes,
#             trans_func=rectify,
#             out_func=softmax,
#             dropout=0.5,
#             conv_dropout=0.5,
#             batch_norm=True,
#             slicers=slicers)

model = BRNN(n_in=(sequence_length, n_features),
             n_hidden=[32, 32],
             n_out=n_classes,
             trans_func=rectify,
             out_func=softmax,
             dropout=0.5,
             bl_dropout=0.5,
             slicers=slicers)

# Copy model to output folder
copy_script(__file__, model)

f_train, f_test, f_validate, train_args, test_args, validate_args, predict = model.build_model(train_set,
                                                                                               test_set,
                                                                                               weights=class_weights)
train_args['inputs']['batchsize'] = batch_size
train_args['inputs']['learningrate'] = 0.003
# test_args['inputs']['batchsize'] = batch_size

train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=None)
train.train_model(f_train, train_args,
                  f_test, test_args,
                  f_validate, validate_args,
                  n_train_batches=n_train_batches,
                  n_test_batches=1,
                  n_epochs=1000,
                  # Any symbolic model variable can be annealed during
                  # training with a tuple of (var_name, every, scale constant, minimum value).
                  anneal=[("learningrate", 50, 0.75, 3e-4)])

# Load test data
annotation_names = json.load(open(data_path + '/annotations.json'))
num_lines = 0
se_cols = ['start', 'end']
seq_len = fs * num_secs
with open('%s/%s_submission.csv' % (model.get_root_path(), pendulum.now('Europe/Copenhagen')), 'w') as fil:
    fil.write(','.join(['record_id'] + se_cols + annotation_names))
    fil.write('\n')

    for te_ind_str in sorted(os.listdir(os.path.join(data_path, 'test'))):
        te_ind = int(te_ind_str)

        meta = json.load(open(os.path.join(data_path, 'test', te_ind_str, 'meta.json')))
        features = pd.read_csv(os.path.join(data_path, 'test', te_ind_str, 'columns_20.csv')).values
        features = features[:meta['end'] * fs]
        features = scaler.transform(features)

        # We pad so all the sequences are the same length
        n_samples, n_features = features.shape
        features = np.concatenate((features, np.zeros(seq_len - n_samples, n_features)), axis=0)
        features = features.reshape((-1, fs, n_features)).astype('float32')

        probs = predict(features)[0]

        starts = range(meta['end'])
        ends = range(1, meta['end'] + 1)

        for start, end, prob in zip(starts, ends, probs):
            row = [te_ind, start, end] + prob.tolist()

            fil.write(','.join(map(str, row)))
            fil.write('\n')

            num_lines += 1

print("{} lines written.".format(num_lines))
