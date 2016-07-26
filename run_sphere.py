import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lasagne.nonlinearities import leaky_rectify, softmax, rectify, elu, very_leaky_rectify
from training.train import TrainModel
from utils import copy_script
from sklearn.cross_validation import train_test_split

from models.sphere_window_convrnn import wconvRNN
from models.sphere_rnn_brnn import HRNN
from models.sphere_brnn import BRNN
from models.sphere_cnn import CNN

import pendulum
import os
import pickle


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
data_ids = range(1, 11)
fs = 20
required_seconds = 29
batch_size = 25
n_features = 29

## Load data
data, targets = load_sequences(data_ids, data_path, fs=fs)

# Scale columns
# scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# Clip the missing columns to 0 and 1
for i in [9, 13, 18, 28]:
    data[:, i] = np.clip(data[:, i], 0, 1)

if True:
    # Sample sequences
    _data = []
    _targets = []
    for i in range((data.shape[0]//(fs*required_seconds))*50):
        # print(data.shape)
        idx = np.random.randint(0, data.shape[0] - required_seconds*fs)
        n_seconds = np.random.randint(10, 29)
        # print(idx, n_seconds)
        sequence = data[idx:idx+n_seconds*fs, :]
        # print(sequence.shape)
        sequence = np.concatenate((sequence, np.zeros((required_seconds * fs - n_seconds*fs, n_features))), axis=0)
        # print(sequence.shape)
        # sequence = sequence.reshape((1, required_seconds * fs, n_features)).astype('float32')
        _data.append(sequence)

        _target = targets[idx//fs:idx//fs + n_seconds, :]
        _target = np.concatenate((_target, np.zeros((required_seconds - n_seconds, 20))))
        _targets.append(_target)

    data = np.concatenate(_data)
    targets = np.concatenate(_targets)

# Calc max seconds
data_lim = (data.shape[0] // (fs * required_seconds)) * (fs * required_seconds)

# Reshape to 29 second windows
data = data[:data_lim].reshape(-1, fs * required_seconds, 29).astype('float32')
targets = targets[:int(data_lim / fs)].reshape(-1, required_seconds, 20)

# Fill up to match batch_size
filling = np.zeros((3, fs * required_seconds, 29))
filling[:, :, [9, 13, 18, 28]] = 1
data = np.concatenate((data, filling))
targets = np.pad(targets, ((0, 3), (0, 0), (0, 0)), mode='constant')

print("Data size")
print(data.shape)
print(targets.shape)

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=100)

print("Training sizes")
print(train_data.shape)
print(train_targets.shape)

print("Testing sizes")
print(test_data.shape)
print(test_targets.shape)

# Set parameters
n_samples, sequence_length, n_features = train_data.shape
n_classes = train_targets.shape[-1]

train_set = (train_data, train_targets)
test_set = (test_data, test_targets)
n_train_batches = train_data.shape[0] // batch_size
n_test_batches = test_data.shape[0] // batch_size

# Slicers for cutting out columns
slicers = dict(pir=slice(0, 10), accel=slice(10, 14), rssi=slice(14, 19), video=slice(19, None))

# Load encoder values
encoder_path = '/home/sdka/dev/deeplearning/output/id_20160725170312_RAE_4_[32]_32/pickled model/encoder.pkl'
enc_values = pickle.load(open(encoder_path, 'rb'))

model = BRNN(n_in=(sequence_length, n_features),
             n_hidden=[32, 32, 32],
             n_out=n_classes,
             n_enc=32,
             enc_values=enc_values,
             freeze_encoder=False,
             trans_func=elu,
             out_func=softmax,
             dropout=0.5,
             bl_dropout=0.5,
             slicers=slicers,
             bn=True)

# Copy model to output folder
copy_script(__file__, model)

f_train, f_test, f_validate, train_args, test_args, validate_args, predict = model.build_model(train_set,
                                                                                               test_set,
                                                                                               weights=class_weights)
train_args['inputs']['batchsize'] = batch_size
train_args['inputs']['learningrate'] = 0.003
# test_args['inputs']['batchsize'] = batch_size

try:
    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=None)
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_train_batches,
                      n_test_batches=1,
                      n_epochs=200,
                      # Any symbolic model variable can be annealed during
                      # training with a tuple of (var_name, every, scale constant, minimum value).
                      anneal=[("learningrate", 10, 0.75, 3e-4)])

except KeyboardInterrupt:
    print('Ending')

finally:

    # Load test data
    annotation_names = json.load(open(data_path + '/annotations.json'))
    num_lines = 0
    se_cols = ['start', 'end']
    seq_len = fs * required_seconds
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
            features = np.concatenate((features, np.zeros((seq_len - n_samples, n_features))), axis=0)
            features = features.reshape((-1, seq_len, n_features)).astype('float32')

            probs = predict(features)[0][0]

            starts = range(meta['end'])
            ends = range(1, meta['end'] + 1)

            for start, end, prob in zip(starts, ends, probs):
                row = [te_ind, start, end] + prob.tolist()

                fil.write(','.join(map(str, row)))
                fil.write('\n')

                num_lines += 1

    print("{} lines written.".format(num_lines))
