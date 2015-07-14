__author__ = 'Simon'
'''
Load data from different Activity Recognition databases.
Each method returns a data dict with train and test data, i.e. 'x_test', 'y_test'
with the shape of n_batch x samples x features
'''

import pandas as pd
import glob as glob
import numpy as np

class LoadHAR(object):
    def __init__(self, root_folder=None):
        self.root_folder = root_folder
        if root_folder is None:
            raise RuntimeError('Invalid folder')

    def uci_har_v1(self):
        sub_folder = '/UCI HAR Dataset v1/'
        test_folder = self.root_folder + sub_folder + 'test/'
        train_folder = self.root_folder + sub_folder + 'train/'

        test_files = glob.glob(test_folder + 'Inertial Signals/body_acc_*')
        train_files = glob.glob(train_folder + 'Inertial Signals/body_acc_*')

        data = dict()
        data['x_test'] = pd.read_csv(test_files[0], sep=r'\s+').values
        for i in range(1, len(test_files)):
            data['x_test'] = np.dstack((data['x_test'], pd.read_csv(test_files[i], sep=r'\s+').values))

        data['x_train'] = pd.read_csv(train_files[0], sep=r'\s+').values
        for i in range(1, len(train_files)):
            data['x_train'] = np.dstack((data['x_train'], pd.read_csv(train_files[i], sep=r'\s+').values))

        data['y_train'] = one_hot(pd.read_csv(train_folder + 'y_train.txt', squeeze=True).values - 1)
        data['y_test'] = one_hot(pd.read_csv(test_folder + 'y_test.txt', squeeze=True).values - 1)

        return data

def one_hot(labels, n_classes=None):
    '''
    Converts an array of label integers to a one-hot matrix encoding
    :parameters:
        - labels : np.ndarray, dtype=int
            Array of integer labels, in {0, n_classes - 1}
        - n_classes : int
            Total number of classes
    :returns:
        - one_hot : np.ndarray, dtype=bool, shape=(labels.shape[0], n_classes)
            One-hot matrix of the input
    '''
    if n_classes is None:
        n_classes = labels.max() + 1

    m = np.zeros((labels.shape[0], n_classes)).astype(bool)
    m[range(labels.shape[0]), labels] = True
    return m
