__author__ = 'Simon'
'''
Load data from different Activity Recognition databases.
Each method returns a data dict with train and test data, i.e. 'x_test', 'y_test'
with the shape of n_batch x samples x features
'''
import theano
theano.config.floatX = 'float32'
import os
import pandas as pd
import glob as glob
import numpy as np
from scipy.io import loadmat
from sklearn.cross_validation import StratifiedShuffleSplit
import itertools
from har_utils import roll, pitch, expand_target

# Path to HAR data
if 'nt' in os.name:
    ROOT_FOLDER = 'D:/PhD/Data/activity/'
else:
    ROOT_FOLDER = '/home/sdka/data/activity/'

class LoadHAR(object):
    def __init__(self, root_folder=ROOT_FOLDER):
        self.root_folder = root_folder
        if root_folder is None:
            raise RuntimeError('Invalid folder')

    def uci_hapt(self, add_pitc=False, add_roll=False):
        subfolder = '/UCI/HAPT Data Set/RawData/'
        files = glob.glob(self.root_folder + subfolder + 'acc_*')

        data = dict()
        for file in files:
            pass

    def uci_har_v1(self, add_pitch=False, add_roll=False, expand=False):
        """
        Data from Ortiz
        :return:
        """
        sub_folder = '/UCI/UCI HAR Dataset v1/'
        test_folder = self.root_folder + sub_folder + 'test/'
        train_folder = self.root_folder + sub_folder + 'train/'

        test_files = sorted(glob.glob(test_folder + 'Inertial Signals/total_acc_*'))
        train_files = sorted(glob.glob(train_folder + 'Inertial Signals/total_acc_*'))

        print("Test files")
        for mfile in test_files:
            print(mfile)

        print("Train files")
        for mfile in train_files:
            print(mfile)

        data = dict()
        data['x_test'] = pd.read_csv(test_files[0], sep=r'\s+').values
        for i in range(1, len(test_files)):
            data['x_test'] = np.dstack((data['x_test'], pd.read_csv(test_files[i], sep=r'\s+').values))

        data['x_train'] = pd.read_csv(train_files[0], sep=r'\s+').values
        for i in range(1, len(train_files)):
            data['x_train'] = np.dstack((data['x_train'], pd.read_csv(train_files[i], sep=r'\s+').values))

        data['y_train'] = one_hot(pd.read_csv(train_folder + 'y_train.txt', squeeze=True).values - 1)
        data['y_test'] = one_hot(pd.read_csv(test_folder + 'y_test.txt', squeeze=True).values - 1)

        if expand:
            data['y_train'] = expand_target(data['y_train'], data['x_test'].shape[1])
            data['y_test'] = expand_target(data['y_test'], data['x_test'].shape[1])

        # Load precomputed features
        features = pd.read_csv(self.root_folder + sub_folder + "/features.txt", header=None, sep=";", names=['features'], squeeze=True).str.strip()
        features_filt = features[features.str.contains(r't\w*Acc')]
        data['x_test_features'] = pd.read_csv(test_folder + 'X_test.txt', sep=r'\s+', names=features)[features_filt].values
        data['x_train_features'] = pd.read_csv(train_folder + 'X_train.txt', sep=r'\s+', names=features)[features_filt].values

        data_mean = np.mean((data['x_test'].mean(), data['x_train'].mean()))
        data_std = np.mean((data['x_test'].std(), data['x_train'].std()))

        print("Data mean: %f, Data std: %f" % (data_mean, data_std))

        # Downsample to 64 samples per window i.e. 25Hz
        ratio = 1
        for key in ['x_test', 'x_train']:
            n_win, n_samp, n_dim = data[key].shape
            if ratio > 1:
                data[key] = downsample(data[key].reshape(-1, n_dim), ratio=ratio).\
                    reshape(n_win, n_samp/ratio, n_dim)

            # Data normalisation
            data[key] = data[key] - data_mean
            data[key] = data[key]/data_std

            # Add pitch and roll
            if add_pitch:
                pitches = []
                for i in range(n_win):
                    pitches.append(pitch(data[key][i]))
                data[key] = np.concatenate((data[key], pitches), axis=2)

            if add_roll:
                rolls = []
                for i in range(n_win):
                    rolls.append(roll(data[key][i,:,:3]))
                data[key] = np.concatenate((data[key], rolls), axis=2)
        # np.savez('data/uci_har_v1_theia.npz', x_train=data['x_train'], y_train=data['y_train'], x_test=data['x_test'], y_test=data['y_test'])
        return return_shared_tuple(data)

    def skoda(self):
        """
        dataset_xx[axis][class][instance] is a vector with calibrated acceleration
        data; length of vector is gesture length dependent and varies with
        class number and instance number.
        output: 64x13
        """
        sub_folder = 'ETH/SkodaMiniCP/'
        data = loadmat(self.root_folder + sub_folder + 'right_classall_clean', squeeze_me=True)['right_classall_clean']
        return data

    def opportunity(self):
        sub_folder = 'UCI/Opportunity/dataset/'
        pass

    def uci_mhealth(self):
        sub_folder = 'UCI/mHealth/'

        # Load the first subject and then the rest iteratively
        tmp = pd.read_csv(self.root_folder + sub_folder + 'mHealth_subject%d.log' % 1,
                          sep='\t',
                          usecols=[0, 1, 2, 23]).values
        for idx in range(2, 11):
            tmp = np.vstack((tmp, pd.read_csv(self.root_folder + sub_folder + 'mHealth_subject%d.log' % idx,
                                         sep='\t',
                                         usecols=[0, 1, 2, 23]).values))
        ratio = 1
        for idx in range(1,13):
            tmp2 = tmp[tmp[:, -1]==idx]
            tmp2 = downsample(tmp2, ratio=ratio)
            tmp2 = window_segment(tmp2, n_samples=64)

            if idx is 1:
                tmp_seg = np.asarray(tmp2)
            else:
                tmp_seg = np.vstack((tmp_seg, np.asarray(tmp2)))

        data = dict()
        for train_index, test_index in StratifiedShuffleSplit(tmp_seg[:, 0, -1],
                                                              n_iter=1,
                                                              test_size=0.2,
                                                              random_state=1):
            data['x_train'], data['x_test'] = tmp_seg[train_index, :, :3], tmp_seg[test_index, :, :3]
            data['y_train'], data['y_test'] = one_hot(tmp_seg[train_index, 0, -1].astype('int')), \
                                              one_hot(tmp_seg[test_index, 0, -1].astype('int'))
        return return_shared_tuple(data)

    def idash(self):
        """
        This dataset contains motion sensor data of 16 physical activities
        (walking, jogging, stair climbing, etc.) collected on 16 adults using
        an iPod touch device (Apple Inc.). The data sampling rate was 30 Hz.
        The collection time for an activity varied from 20 seconds to 17 minutes.
        """
        sub_folder = 'Physical Activity Sensor Data-Public/Public/iDASH_activity_dataset/'
        subjects = range(1, 17)
        n_samples=60
        cols = [0, 1, 2]
        tmp_seg = np.empty((0, n_samples, len(cols)))
        target = []
        for subject in subjects:
            files = sorted(glob.glob(self.root_folder + sub_folder + '%d/*' % subject))
            for idx, csv_file in enumerate(files):
                if not "blank" in csv_file:
                    tmp = pd.read_csv(csv_file, sep=',', usecols=cols).values
                    tmp = window_segment(tmp, n_samples=n_samples)
                    tmp_seg = np.vstack((tmp_seg, tmp))
                    target.append(np.ones((tmp.shape[0],), dtype=np.int)*idx)
        target = np.asarray(list(itertools.chain.from_iterable(target)))

        data = dict()
        for train_index, test_index in StratifiedShuffleSplit(target,
                                                              n_iter=1,
                                                              test_size=0.2,
                                                              random_state=1):
            data['x_train'], data['x_test'] = tmp_seg[train_index], tmp_seg[test_index]
            data['y_train'], data['y_test'] = one_hot(target[train_index].astype('int')), \
                                              one_hot(target[test_index].astype('int'))
        return return_shared_tuple(data)

    def pamap2(self):
        pass

    def mhealth_maninni(self):
        """
        Data from Stanford/MIT. Unfortunately they only provide a magnitude vector
        Subjects: 33
        """
        sub_folder = 'mhealth/Mannini_Data_and_Code_PMC2015/Data/'
        data = loadmat(sorted(glob.glob(self.root_folder + sub_folder + '*.mat'))[0])
        X = data['Data_m']

def one_hot(labels, n_classes=None):
    """
    Converts an array of label integers to a one-hot matrix encoding
    :parameters:
        - labels : np.ndarray, dtype=int
            Array of integer labels, in {0, n_classes - 1}
        - n_classes : int
            Total number of classes
    :returns:
        - one_hot : np.ndarray, dtype=bool, shape=(labels.shape[0], n_classes)
            One-hot matrix of the input
    """
    if n_classes is None:
        n_classes = labels.max() + 1

    m = np.zeros((labels.shape[0], n_classes)).astype(bool)
    m[range(labels.shape[0]), labels] = True
    return m

def downsample(data, ratio=2):
    # Downsample data with ratio
    n_samp, n_dim = data.shape

    # New number of samples
    n_resamp = n_samp/ratio

    # Reshape, mean and shape back
    data = np.reshape(data[:n_resamp*ratio], (n_resamp, ratio, n_dim))
    data = np.mean(data, axis=1)
    return data

def window_segment(data, n_samples = 64):
    # Segment in windows on axis 1
    n_samp, n_dim = data.shape
    n_win = n_samp/n_samples
    data = np.reshape(data[:n_win*n_samples], (n_win, n_samples, n_dim))
    return data

def shared_dataset(data_xy, borrow=False):
    x, y = data_xy
    shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, shared_y

def return_shared_tuple(data):
    n_train, sequence_length, n_features = data['x_train'].shape
    n_classes = data['y_train'].shape[-1]
    print('Sequence: %d' % sequence_length)
    print('Features: %d' % n_features)
    print('Classes: %d' % n_classes)
    return shared_dataset((data['x_train'].astype(theano.config.floatX), data['y_train'].astype(theano.config.floatX))), \
           shared_dataset((data['x_test'].astype(theano.config.floatX), data['y_test'].astype(theano.config.floatX))), \
           shared_dataset((data['x_test'].astype(theano.config.floatX), data['y_test'].astype(theano.config.floatX))), \
           (int(sequence_length), int(n_features), int(n_classes))