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
from scipy import interpolate
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
import itertools
from har_utils import roll, pitch, expand_target, split_signal, magnitude, rolling_window
from sklearn import preprocessing
import cPickle as pickle

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
        self.name = ""

    def uci_hapt(self, add_pitch=False, add_roll=False, expand=False, add_filter=False, n_samples=200, step=200, shuffle=False):
        """
        Sampling rate = 50
        :param add_pitc:
        :param add_roll:
        :param expand:
        :param add_filter:
        :param n_samples: number of samples in one window
        :param step: step between windows. step < n_samples creates overlap
        :return: dict of train, test data
        """
        self.name = "UCI HAPT"
        subfolder = 'UCI/HAPT Data Set/RawData/'
        data_file = self.root_folder+subfolder+'/data.npz'

        if os.path.isfile(data_file):
            data = pickle.load(open(data_file, 'r'))
        else:
            files = sorted(glob.glob(self.root_folder + subfolder + 'acc_*'))
            labels = pd.read_csv(self.root_folder + subfolder +'/labels.txt',
                                 names=['exp', 'user', 'activity', 'start', 'end'],
                                 header=None, sep=' ')

            # Extract signals from the files and split them into segments. UCI HAR V1 uses 128 window length with
            # a step size of 64
            data_array = np.empty((0, n_samples, 3))
            y = np.empty((0))
            for exp, user in labels[['exp', 'user']].drop_duplicates().values:
                print("Loading %s" % self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user))
                values = pd.read_csv(self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user), sep=' ').values
                idx = ((labels['exp']==exp) & (labels['user']==user))

                for activity, start, end in labels[['activity', 'start', 'end']][idx].values:
                    segment = values[start:end]
                    # Pad a segment if it is smaller than windows size
                    if segment.shape[0] < n_samples:
                        segment = np.pad(segment, ((0, n_samples - segment.shape[0]), (0, 0)), 'edge')
                    # segment = window_segment(segment, window=n_samples)
                    segment = rolling_window(segment, (n_samples, 0), step).swapaxes(1, 2)
                    data_array = np.concatenate((data_array, segment))
                    y = np.concatenate((y, [(activity)]*segment.shape[0]))
            # for exp, user in labels[['exp', 'user']].drop_duplicates().values:
            #     print("Loading %s" % self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user))
            #     df = pd.read_csv(self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user), sep=' ')
            #     idx = ((labels['exp']==exp) & (labels['user']==user))
            #
            #     # Initialize activity column to zeros
            #     df['activity'] = 0
            #     for activity, start, end in labels[['activity', 'start', 'end']][idx].values:
            #         df['activity'].loc[start:end] = activity
            #
            #     # Remove samples without label
            #     df = df[df['activity'] != 0]
            #
            #     # Segment into windows with overlap
            #     segmented = rolling_window(df.values, (n_samples, 0), step).swapaxes(1, 2)
            #
            #     # Find y label
            #     t = []
            #     for idx in range(segmented.shape[0]):
            #         t.append(np.argmax(np.bincount(segmented[idx, :, -1].astype('int'))))
            #
            #     # Collect data
            #     y = np.concatenate((y, t))
            #     data_array = np.concatenate((data_array, segmented[:, :, :-1]))
            print('Data shape:', data_array.shape)
            print('Target shape:', y.shape)
            print('Unique targets: %d' % np.count_nonzero(np.unique(y.flatten()).astype('int')))


            if expand:
                print("Expanding targets")
                y = expand_target(y, data_array.shape[1])

            # Add features to data
            enrich_fs = 0
            if add_filter: enrich_fs = 50
            data_array = add_features(data_array, normalise=False, add_roll=add_roll, add_pitch=add_pitch, expand=expand, enrich_fs=enrich_fs)

            # Standardize data
            n_windows, _, n_features = data_array.shape
            data_array = preprocessing.scale(data_array.reshape((-1, n_features))).reshape((n_windows, n_samples, n_features))

            # Partition data. UCI HAR V1 uses 70/30 split
            data = dict()
            if shuffle:
                for train_index, test_index in StratifiedShuffleSplit(y,
                                                                      n_iter=1,
                                                                      test_size=0.3,
                                                                      random_state=None):
                    data['x_train'], data['x_test'] = data_array[train_index], data_array[test_index]
                    data['y_train'], data['y_test'] = one_hot(y[train_index].astype('int') - 1), \
                                                      one_hot(y[test_index].astype('int') - 1)
            else:
                train_index, test_index = slice(0, np.ceil(0.7*n_windows).astype('int')), \
                                          slice(np.ceil(0.7*n_windows).astype('int')+1, n_windows)
                data['x_train'], data['x_test'] = data_array[train_index], data_array[test_index]
                data['y_train'], data['y_test'] = one_hot(y[train_index].astype('int') - 1), \
                                                  one_hot(y[test_index].astype('int') - 1)

            # Save to disk
            # pickle.dump(data, open(data_file,"w"))

        return return_tuple(data, 12), self.name

    def uci_har_v1(self, add_pitch=False, add_roll=False, expand=False, add_filter=False):
        """
        Data from Ortiz
        Sampling rate: 50hz
        Data is split into training and test as follows: "The obtained dataset has been randomly partitioned into
        two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data."
        :return:
        """
        self.name = "UCI HAR V1"
        sub_folder = 'UCI/UCI HAR Dataset v1/'
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

        # Load precomputed features
        features = pd.read_csv(self.root_folder + sub_folder + "/features.txt", header=None, sep=";", names=['features'], squeeze=True).str.strip()
        features_filt = features[features.str.contains(r't\w*Acc')]
        data['x_test_features'] = pd.read_csv(test_folder + 'X_test.txt', sep=r'\s+', names=features)[features_filt].values
        data['x_train_features'] = pd.read_csv(train_folder + 'X_train.txt', sep=r'\s+', names=features)[features_filt].values

        enrich_fs = 0
        if add_filter: enrich_fs = 50
        data = add_features_dict(data, normalise=True, ratio=1, add_roll=add_roll, add_pitch=add_pitch, expand=expand, enrich_fs=enrich_fs)

        # np.savez('data/uci_har_v1_theia.npz', x_train=data['x_train'], y_train=data['y_train'], x_test=data['x_test'], y_test=data['y_test'])
        return return_tuple(data, 6), self.name

    def wisdm(self, add_pitch=False, add_roll=False, expand=False, add_filter=False):
        """
        Sampling rate: 20hz
        User: 1-36
        Activity: Walking, jogging, sitting, standing, upstairs, downstairs
        :return: shared tuple of data
        """
        self.name = "WISDM"
        sub_folder = 'WISDM Actitracker/Lab/'
        filename = 'WISDM_ar_v1.1_raw.txt'
        columns = ['user','activity','timestamp','x','y','z']
        df = pd.read_csv(self.root_folder + sub_folder + filename, names=columns, lineterminator=';')
        df = df.dropna()
        activity_map = {'Walking':0, 'Jogging':7, 'Sitting':4, 'Standing':5, 'Upstairs':2, 'Downstairs':3}
        df['activity'] = df['activity'].apply(lambda x: activity_map[x])

        # Normalise data
        df_tmp = df[['x','y','z']]
        df[['x','y','z']] = (df_tmp - df_tmp.mean().mean()) / df_tmp.std()

        n_samples = 50
        tmp = np.empty((0, n_samples, 4))
        for user in df['user'].unique():
            tmp = np.concatenate((tmp,
                                 window_segment(df[['x', 'y', 'z','activity']][df['user'] == user].values,
                                                window=n_samples)), axis=0)

        data = dict()
        for train_index, test_index in StratifiedShuffleSplit(tmp[:, 0, -1],
                                                              n_iter=1,
                                                              test_size=0.2,
                                                              random_state=1):
            data['x_train'], data['x_test'] = tmp[train_index, :, :3], tmp[test_index, :, :3]
            data['y_train'], data['y_test'] = one_hot(tmp[train_index, 0, -1].astype('int')), \
                                              one_hot(tmp[test_index, 0, -1].astype('int'))

        enrich_fs = 0
        if add_filter: enrich_fs = 50
        data = add_features(data, normalise=True, ratio=1, add_roll=add_roll, add_pitch=add_pitch, expand=expand, enrich_fs=enrich_fs)

        return return_tuple(data), self.name

    def lingacceleration(self):
        """
        Sampling rate: 76Hz
        actlist.mat contains list of 20 activities
        Data is stored as agg_s<subject no.>_<activity/obstacle>_<date>.mat in either
        'activity'(lab) or 'obstacle'(natural)
        actdata{i,j} contains acceleration samples for the activity label actlist{i} from the jth accelerometer source.
        The indexing order of the sources is as follows:
        j = 1 corresponds to data from the right hip,
        2 is for data from the dominant wrist,
        3 is for data from the non-dominant arm,
        4 is for data from the dominant ankle,
        5 is for data from the non-dominant thigh.

        acttime{i,j} contains hoarder timestamps for actdata{i,j}.  There is one timestamp for every 100 samples of
        accelerometer data.
        celldata{j} contains raw acceleration data from accelerometer source j.
        celltime{j} contains timestamps for celldata{j}.
        acts(i) is the ith activity the subject performed in chronological order during the study.
        times(i) specifies the index into celldata{j} of acts(i).  For example, accelerometer data from source j for
        acts(1) corresponds to data from celldata{j}(times(1):times(2)-1).
        freqs(j) contains the average sampling frequency of hoarder source j in Hz.
        starttime contains the start time for the study.  The vector specifies the year, month, date, hour, minute,
        and second in that order the subject began the study.
        :return: tuple
        """
        RuntimeError('Only provides two axes')
        sub_folder = 'mhealth/lingacceleration/'
        activities = [a[0] for a in loadmat('actlist.mat')['actlist'][0]]

        for mat_file in sorted(glob.glob(self.root_folder + sub_folder + 'dataset/activity/*.mat')):
            pass

    def skoda(self):
        """
        dataset_xx[axis][class][instance] is a vector with calibrated acceleration
        data; length of vector is gesture length dependent and varies with
        class number and instance number.
        output: 64x13
        """
        RuntimeError('Not implemented')
        sub_folder = 'ETH/SkodaMiniCP/'
        data = loadmat(self.root_folder + sub_folder + 'right_classall_clean', squeeze_me=True)['right_classall_clean']
        return data

    def opportunity(self):
        RuntimeError('Not implementet')
        sub_folder = 'UCI/Opportunity/dataset/'
        pass

    def uci_mhealth(self, add_pitch=False, add_roll=False, expand=False):
        self.name = "UCI mHealth"
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
            tmp2 = window_segment(tmp2, window=64)

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

        data = add_features(data, True, ratio, add_roll, add_pitch, expand)
        return return_tuple(data), self.name

    def idash(self, add_pitch=False, add_roll=False, expand=False):
        """
        This dataset contains motion sensor data of 16 physical activities
        (walking, jogging, stair climbing, etc.) collected on 16 adults using
        an iPod touch device (Apple Inc.). The data sampling rate was 30 Hz.
        The collection time for an activity varied from 20 seconds to 17 minutes.
        """
        self.name = "IDASH"
        sub_folder = 'Physical Activity Sensor Data-Public/Public/iDASH_activity_dataset/'
        subjects = range(1, 17)
        n_samples = 60
        cols = [0, 1, 2]
        tmp_seg = np.empty((0, n_samples, len(cols)))
        target = []
        for subject in subjects:
            files = sorted(glob.glob(self.root_folder + sub_folder + '%d/*' % subject))
            for idx, csv_file in enumerate(files):
                if not "blank" in csv_file:
                    tmp = pd.read_csv(csv_file, sep=',', usecols=cols).values
                    tmp = window_segment(tmp, window=n_samples)
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

        data = add_features(data, True, 1, add_roll, add_pitch, expand, 30)
        return return_tuple(data), self.name

    def pamap2(self):

        pass

    def mhealth_maninni(self):
        """
        Data from Stanford/MIT. Unfortunately they only provide a magnitude vector
        Subjects: 33
        """
        RuntimeError('Only provides magnitude signal')
        sub_folder = 'mhealth/Mannini_Data_and_Code_PMC2015/Data/'
        file_name = 'StanfordDataset2010_adult_5sensors_win10000_corrected_classes1to3_apr2013_90Hz_filt_o4f20_recalib_PMC.mat'
        data = loadmat(self.root_folder + sub_folder + file_name)
        activities = data['FeatureNames']
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
    data = np.reshape(data[:n_resamp*ratio], (n_resamp, ratio, n_dim)).mean(axis=1)
    return data

def window_segment(data, window = 64):
    # Segment in windows on axis 1
    n_samp, n_dim = data.shape
    n_win = n_samp//(window)
    data = np.reshape(data[:n_win * window], (n_win, window, n_dim))
    return data

def add_features(data, normalise=True, ratio=0, add_roll=False, add_pitch=False, expand=False, enrich_fs=0):

    if normalise:
        # data_mag = np.mean((magnitude(data['x_test']), magnitude(data['x_train'])))
        data_mean = data.mean()
        data_std = data.reshape(-1, 3).std(axis=0)
        print("Data mean: %f, Data std: %f" % (data_mean, data_std.mean()))

    n_win, n_samp, n_dim = data.shape
    if ratio > 1:
        data = downsample(data.reshape(-1, n_dim), ratio=ratio).\
            reshape(n_win, n_samp/ratio, n_dim)

    if normalise:
        # Data normalisation
        data = data - data_mean
        data = data/data_std

    # Add pitch and roll
    if add_pitch:
        pitches = []
        for i in range(n_win):
            pitches.append(pitch(data[i]))
        data = np.concatenate((data, pitches), axis=2)

    if add_roll:
        rolls = []
        for i in range(n_win):
            rolls.append(roll(data[i,:,:3]))
        data = np.concatenate((data, rolls), axis=2)

    if enrich_fs>0:
        tmp_lp = split_signal(data[:,:,:3], enrich_fs)
        tmp_hp = data[:,:,:3] - tmp_lp
        data = np.concatenate((data, tmp_lp, tmp_hp), axis=2)

    return data

def add_features_dict(data, normalise=True, ratio=0, add_roll=False, add_pitch=False, expand=False, enrich_fs=0):
    """
    Concatenate features on the last dimension
    """
    for key in ['x_test', 'x_train']:
        n_win, n_samp, n_dim = data[key].shape
        if ratio > 1:
            data[key] = downsample(data[key].reshape(-1, n_dim), ratio=ratio).\
                reshape(n_win, n_samp/ratio, n_dim)

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

        if enrich_fs>0:
            tmp_lp = split_signal(data[key][:,:,:3], enrich_fs)
            tmp_hp = data[key][:,:,:3] - tmp_lp
            data[key] = np.concatenate((data[key], tmp_lp, tmp_hp), axis=2)

    if normalise:
        n_dim = data['x_test'].shape[2]
        # data_mag = np.mean((magnitude(data['x_test']), magnitude(data['x_train'])))
        data_mean = np.mean((data['x_test'].mean(), data['x_train'].mean()))
        data_std = np.mean((data['x_test'].reshape(-1, n_dim).std(axis=0),
                            data['x_train'].reshape(-1, n_dim).std(axis=0)),
                           axis=0)
        print("Data mean: %f, Data std: %f" % (data_mean, data_std.mean()))

        for key in ['x_test', 'x_train']:
            # Data normalisation of each feature
            data[key] = data[key] - data_mean
            data[key] = data[key]/data_std

    if expand:
        print("Expanding targets")
        data['y_train'] = expand_target(data['y_train'], data['x_test'].shape[1])
        data['y_test'] = expand_target(data['y_test'], data['x_test'].shape[1])

    return data


def shared_dataset(data_xy, borrow=False):
    x, y = data_xy
    shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, shared_y

def return_tuple(data, n_classes):
    n_train, sequence_length, n_features = data['x_train'].shape
    print('Sequence: %d' % sequence_length)
    print('Features: %d' % n_features)
    print('Classes: %d' % n_classes)
    return (data['x_train'], data['y_train']), \
           (data['x_test'], data['y_test']), \
           (data['x_test'], data['y_test']), \
           (int(sequence_length), int(n_features), int(n_classes))