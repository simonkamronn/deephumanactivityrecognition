import theano
from training.train import TrainModel
from lasagne_extensions.nonlinearities import rectify, leaky_rectify
from models.rae import RAE
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import copy_script
import os

seed = np.random.randint(1, 2147462579)
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
num_secs = 29
batch_size = 100

# Load data
data, targets = load_sequences(data_ids, data_path, fs=fs)

# Load test data
for te_ind_str in sorted(os.listdir(os.path.join(data_path, 'test'))):
    te_ind = int(te_ind_str)

    meta = json.load(open(os.path.join(data_path, 'test', te_ind_str, 'meta.json')))
    features = pd.read_csv(os.path.join(data_path, 'test', te_ind_str, 'columns_20.csv')).values
    features = features[:meta['end'] * fs]
    data = np.concatenate((data, features))

# Scale columns
# scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# Clip the missing columns to 0 and 1
for i in [9, 13, 18, 28]:
    data[:, i] = np.clip(data[:, i], 0, 1)

# Calc max seconds
data_lim = (data.shape[0] // fs) * fs

# Reshape to 1 second windows
data = data[:data_lim].reshape(-1, fs, 29).astype('float32')
targets = targets[:int(data_lim / fs)].reshape(-1, 1, 20)

print("Data size")
print(data.shape)
print(targets.shape)

# Set parameters
n_samples, sequence_length, n_features = data.shape
n_classes = targets.shape[-1]

# Slicers for cutting out columns
slicers = dict(pir=slice(0, 10), accel=slice(10, 14), rssi=slice(14, 19), video=slice(19, None))

# Slice out modalities
dd = dict()
for key, val in slicers.items():
    dd[key] = data[:, :, val]

data = dd['accel']

# Split into train and test stratified by users
train_data, test_data = train_test_split(data, test_size=0.1)  # , stratify=np.argmax(targets, axis=1), random_state=1)

# Combine in sets
train_set = (data, [])
test_set = (test_data, [])
print('Train size: ', train_set[0].shape)
print('Test size: ', test_set[0].shape)

n, n_l, n_c = train_set[0].shape
n_batches = n // batch_size

# Initialize recurrent variational autoencoder
model = RAE(n_c=int(n_c), n_l=int(n_l), px_hid=[32], enc_rnn=32, dec_rnn=32,
            nonlinearity=leaky_rectify, batchnorm=False)

# Copy model to output folder
copy_script(__file__, model)

# Get the training functions.
f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(train_set, test_set)

# Update the default function arguments.
train_args['inputs']['batchsize'] = batch_size
train_args['inputs']['learningrate'] = 0.005
train_args['inputs']['beta1'] = 0.9
train_args['inputs']['beta2'] = 0.999


def custom_evaluation(model, path):
    # Dump encoder
    model.save_encoder()

    # Get model output
    x_ = test_set[0]
    xhat = model.f_px(x_)

    idx = np.random.randint(0, x_.shape[0]-5)

    plt.clf()
    f, axarr = plt.subplots(nrows=1, ncols=1)
    axarr.plot(x_[idx:idx+5, :, :].reshape(-1, x_.shape[-1]), color='red')
    axarr.plot(xhat[idx:idx+5, :, :].reshape(-1, x_.shape[-1]), color='blue', linestyle='dotted')

    f.set_size_inches(12, 10)
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
                  n_epochs=5000,
                  anneal=[("learningrate", 100, 0.75, 3e-5)])