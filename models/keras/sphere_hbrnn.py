from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Lambda, Layer, Input, Reshape, TimeDistributed
from keras.optimizers import RMSprop
from keras.activations import relu, softmax
import keras.backend as K
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import normalize, MinMaxScaler

GRAD_CLIP = 5


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
train_ids = range(1, 11)
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




