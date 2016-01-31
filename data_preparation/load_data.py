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
from scipy.signal import resample
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
import itertools
from har_utils import roll, pitch, expand_target, split_signal, magnitude, rolling_window, lowpass_filter
from sklearn import preprocessing
import cPickle as pickle

ACTIVITY_MAP = {0: 'WALKING', 1: 'CYCLING', 2: 'RUNNING', 3: 'STAIRS', 4: 'JOGGING', 5: 'LAYING',
                6: 'WALKING_UPSTAIRS',7: 'WALKING_DOWNSTAIRS', 8: 'BEND_FORWARD',
                9: 'ARM_ELEVATION', 10: 'KNEE_BEND', 11: 'SITTING', 12: 'STANDING', 13: 'JUMP', 14: 'STEP', 15: 'NULL',
                16: 'UPRIGHT_INACTIVE', 17: 'INACTIVE',
                18: 'STAND_TO_SIT', 19: 'SIT_TO_STAND', 20: 'SIT_TO_LIE', 21: 'LIE_TO_SIT', 22: 'STAND_TO_LIE',
                23: 'LIE_TO_STAND'}
MAP_ACTIVITY = dict((v, k) for k, v in ACTIVITY_MAP.iteritems())
SR = 50

# Path to HAR data
if 'nt' in os.name:
    ROOT_FOLDER = 'D:/PhD/Data/activity/'
else:
    ROOT_FOLDER = '/nobackup/titans/sdka/data/activity/'

class LoadHAR(object):
    def __init__(self, root_folder=ROOT_FOLDER, add_pitch=False, add_roll=False, expand=False,
                 add_filter=False, n_samples=200, step=200, normalize=True, comp_magnitude=False,
                 simple_labels=False, common_labels=False):
        self.root_folder = root_folder
        if root_folder is None:
            raise RuntimeError('Invalid folder')
        self.name = ""

        self.add_pitch = add_pitch
        self.add_roll = add_roll
        self.expand = expand
        self.add_filter = add_filter
        self.n_samples = n_samples
        self.step = step
        self.normalize = normalize
        self.comp_magnitude = comp_magnitude
        self.simple_labels = simple_labels
        self.common_labels = common_labels

    def uci_hapt(self):
        """
        Sampling rate = 50
        1 WALKING
        2 WALKING_UPSTAIRS
        3 WALKING_DOWNSTAIRS
        4 SITTING
        5 STANDING
        6 LAYING
        7 STAND_TO_SIT
        8 SIT_TO_STAND
        9 SIT_TO_LIE
        10 LIE_TO_SIT
        11 STAND_TO_LIE
        12 LIE_TO_STAND
        """
        self.name = "UCI HAPT"
        subfolder = 'UCI/HAPT Data Set/RawData/'
        data_file = self.root_folder+subfolder+'/data.npz'
        if not self.simple_labels:
            activity_map = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING',
                            5: 'STANDING', 6: 'LAYING', 7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND',
                            9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT', 11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND'}
        else:
            activity_map = {1: 'WALKING', 2: 'STAIRS', 3: 'STAIRS', 4: 'INACTIVE',
                            5: 'INACTIVE', 6: 'INACTIVE', 7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND',
                            9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT', 11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND'}
        if os.path.isfile(data_file):
            data = pickle.load(open(data_file, 'r'))
        else:
            files = sorted(glob.glob(self.root_folder + subfolder + 'acc_*'))
            labels = pd.read_csv(self.root_folder + subfolder +'/labels.txt',
                                 names=['exp', 'user', 'activity', 'start', 'end'],
                                 header=None, sep=' ')

            # Extract signals from the files and split them into segments. UCI HAR V1 uses 128 window length with
            # a step size of 64
            data_array = np.empty((0, self.n_samples, 3))
            y = np.empty((0))
            users = np.empty((0))

            # for exp, user in labels[['exp', 'user']].drop_duplicates().values:
            #     print("Loading %s" % self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user))
            #     values = pd.read_csv(self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user), sep=' ').values
            #     idx = ((labels['exp']==exp) & (labels['user']==user))
            #
            #     for activity, start, end in labels[['activity', 'start', 'end']][idx].values:
            #         segment = values[start:end]
            #
            #         # Pad a segment to a multiple of n_samples
            #         if segment.shape[0] <= self.n_samples:
            #             pad_width = int(np.ceil(segment.shape[0]/float(self.n_samples))*self.n_samples - segment.shape[0])
            #             segment = np.pad(segment, ((0, pad_width), (0, 0)), 'edge').reshape(1, self.n_samples, 3)
            #
            #         # Segment with a rolling window allowing overlap
            #         else:
            #             try:
            #                 segment = rolling_window(segment, (self.n_samples, 0), self.step).swapaxes(1, 2)
            #             except ValueError, e:
            #                 print(e)
            #                 print(segment.shape)
            #
            #         # Collect data
            #         data_array = np.concatenate((data_array, segment))
            #         y = np.concatenate((y, [activity]*segment.shape[0]))
            #         users = np.concatenate((users, [(user)]*segment.shape[0]))
            for exp, user in labels[['exp', 'user']].drop_duplicates().values:
                print("Loading %s" % self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user))
                df = pd.read_csv(self.root_folder + subfolder + 'acc_exp%02d_user%02d.txt' % (exp, user), sep=' ')
                idx = ((labels['exp']==exp) & (labels['user']==user))

                values = df.values
                activities = np.zeros((values.shape[0], 1))
                for activity, start, end in labels[['activity', 'start', 'end']][idx].values:
                    activities[start:end] = activity

                values = np.concatenate((values, activities), axis=1)

                # Segment into windows with overlap
                segmented = rolling_window(values, (self.n_samples, 0), self.step).swapaxes(1, 2)

                # Find y label
                # t = []
                # for idx in range(segmented.shape[0]):
                    # t.append(segmented[idx, -1, -1].astype('int'))
                t = np.asarray(segmented[:, -1, -1]).astype('int')

                # Remove samples without label
                idx = t != 0
                segmented = segmented[idx]
                t = t[idx]

                # Collect data
                y = np.concatenate((y, t))
                data_array = np.concatenate((data_array, segmented[:, :, :-1]))
                users = np.concatenate((users, [(user)]*len(t)))
        print('Data shape:', data_array.shape)
        print('Target shape:', y.shape)
        print('Unique targets: %d' % np.count_nonzero(np.unique(y.flatten()).astype('int')))

        if self.expand:
            print("Expanding targets")
            y = expand_target(y, data_array.shape[1])

        # Add features to data
        data_array, stats = self.add_features(data_array,
                                              normalise=self.normalize,
                                              add_roll=self.add_roll,
                                              add_pitch=self.add_pitch,
                                              add_filter=self.add_filter,
                                              comp_magnitude=self.comp_magnitude)

        # Convert to common labels
        if self.common_labels:
            y = self.map_to_common_activities(y, activity_map)
        else:
            y = y - 1

        # Save to disk
        # pickle.dump(data, open(data_file,"w"))

        return data_array, y.astype('int'), self.name, users, stats

    def uci_har_v1(self):
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
        activity_map = {1: 'WALKING', 2: 'STAIRS',3: 'STAIRS', 4: 'INACTIVE',
                        5: 'INACTIVE', 6: 'INACTIVE', 7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND',
                        9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT', 11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND'}

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

        data['y_train'] = (pd.read_csv(train_folder + 'y_train.txt', squeeze=True).values - 1)
        data['y_test'] = (pd.read_csv(test_folder + 'y_test.txt', squeeze=True).values - 1)

        users = pd.read_csv(train_folder + 'subject_train.txt', squeeze=True).values
        users = np.concatenate((users, pd.read_csv(test_folder + 'subject_test.txt', squeeze=True).values))

        # Load precomputed features
        # features = pd.read_csv(self.root_folder + sub_folder + "/features.txt", header=None, sep=";", names=['features'], squeeze=True).str.strip()
        # features_filt = features[features.str.contains(r't\w*Acc')]
        # data['x_test_features'] = pd.read_csv(test_folder + 'X_test.txt', sep=r'\s+', names=features)[features_filt].values
        # data['x_train_features'] = pd.read_csv(train_folder + 'X_train.txt', sep=r'\s+', names=features)[features_filt].values

        data = np.concatenate((data['x_train'], data['x_test']), axis=0)
        y = np.concatenate((data['y_train'], data['y_test']), axis=0)

        data_array, stats = self.add_features(data_array,
                                       normalise=self.normalize,
                                       add_roll=self.add_roll,
                                       add_pitch=self.add_pitch,
                                       add_filter=self.add_filter,
                                       comp_magnitude=self.comp_magnitude)
        # Convert to common labels
        y = self.map_to_common_activities(y, activity_map)

        # np.savez('data/uci_har_v1_theia.npz', x_train=data['x_train'], y_train=data['y_train'], x_test=data['x_test'], y_test=data['y_test'])
        return data_array, y.astype('int'), self.name, users, stats

    def wisdm1(self):
        """
        Sampling rate: 20hz
        User: 1-36
        Activity: Walking, jogging, sitting, standing, upstairs, downstairs
        :return: shared tuple of data
        """
        self.name = "WISDM1"
        sub_folder = 'WISDM/Lab/'
        filename = 'WISDM_ar_v1.1_raw.txt'
        columns = ['user','labels','timestamp','x','y','z']
        dtypes = {'user': np.int32,'labels': np.str,'timestamp': np.float64,'x': np.float32,'y': np.float32,'z': np.float32}
        df = pd.read_csv(self.root_folder + sub_folder + filename, names=columns, lineterminator=';', dtype=dtypes)
        df = df.dropna()
        activity_map = {'Walking': 'WALKING', 'Upstairs': 'STAIRS', 'Stairs': 'STAIRS', 'Downstairs': 'STAIRS',
                        'Sitting': 'INACTIVE', 'Standing': 'INACTIVE', 'LyingDown': 'INACTIVE', 'Jogging': 'JOGGING'}
        df['labels'] = df['labels'].apply(activity_map.get)
        df['labels'] = df['labels'].apply(MAP_ACTIVITY.get)
        sr = 20.

        tmp = np.empty((0, 5))
        for user in df['user'].unique():
            tmp = np.concatenate((tmp, df[['x', 'y', 'z','labels','user']][df['user'] == user].values), axis=0)

        # window samples into the equivalent of n_samples at 50 Hz.
        tmp = rolling_window(tmp, window=(self.n_samples * sr/SR, 0), step=self.step * sr/SR).swapaxes(1, 2)
        n_windows, sequence_length, _ = tmp.shape
        y = np.zeros(n_windows)
        users = np.zeros(n_windows)
        data_array = np.empty((n_windows, self.n_samples, 3))
        for idx in range(n_windows):
            y[idx] = np.argmax(np.bincount(tmp[idx, :, -2].astype('int')))
            users[idx] = np.argmax(np.bincount(tmp[idx, :, -1].astype('int')))
            for f in range(3):
                data_array[idx, :, f] = resample(tmp[idx, :, f], self.n_samples)

        data_array, stats = self.add_features(data_array,
                                       normalise=self.normalize,
                                       ratio=0,
                                       add_roll=self.add_roll,
                                       add_pitch=self.add_pitch,
                                       add_filter=self.add_filter,
                                       comp_magnitude=self.comp_magnitude)
        return data_array, y.astype('int'), self.name, users, stats

    def wisdm2(self):
        """
        Sampling rate: 20hz
        Activity: Walking, jogging, sitting, standing, upstairs, downstairs
        :return: tuple of data
        """
        self.name = "WISDM2"
        sub_folder = 'WISDM/Real/'
        filename = 'WISDM_at_v2.0_raw.txt'
        columns = ['user','labels','timestamp','x','y','z']
        dtypes = {'user': np.int32,'labels': np.str,'timestamp': np.float64,'x': np.float32,'y': np.float32,'z': np.float32}
        df = pd.read_csv(self.root_folder + sub_folder + filename, names=columns, lineterminator=';', dtype=dtypes)
        df = df.dropna()
        activity_map = {'Walking': 'WALKING', 'Upstairs': 'STAIRS', 'Stairs': 'STAIRS', 'Downstairs': 'STAIRS',
                        'Sitting': 'INACTIVE', 'Standing': 'INACTIVE', 'LyingDown': 'INACTIVE', 'Jogging': 'JOGGING'}
        df['labels'] = df['labels'].apply(activity_map.get)
        df['labels'] = df['labels'].apply(MAP_ACTIVITY.get)
        sr = 20.

        tmp = np.empty((0, 5))
        for user in df['user'].unique():
            tmp = np.concatenate((tmp, df[['x', 'y', 'z','labels','user']][df['user'] == user].values), axis=0)

        # window samples into the equivalent of n_samples at 50 Hz.
        tmp = rolling_window(tmp, window=(self.n_samples * sr/SR, 0), step=self.step * sr/SR).swapaxes(1, 2)
        n_windows, sequence_length, n_features = tmp.shape
        y = np.zeros(n_windows)
        users = np.zeros(n_windows)
        data_array = np.empty((n_windows, self.n_samples, 3))
        for idx in range(n_windows):
            y[idx] = np.argmax(np.bincount(tmp[idx, :, -2].astype('int')))
            users[idx] = np.argmax(np.bincount(tmp[idx, :, -1].astype('int')))
            for f in range(3):
                data_array[idx, :, f] = resample(tmp[idx, :, f], self.n_samples)

        data_array = self.add_features(data_array,
                                       normalise=self.normalize,
                                       ratio=0,
                                       add_roll=self.add_roll,
                                       add_pitch=self.add_pitch,
                                       add_filter=self.add_filter,
                                       comp_magnitude=self.comp_magnitude)
        return data_array, y.astype('int'), self.name, users

    def uci_mhealth(self):
        '''
        #Activities: 12
        #Sensor devices: 3 (chest, left ankel, right arm)
        #Subjects: 10
        L1: Standing still (1 min)
        L2: Sitting and relaxing (1 min)
        L3: Lying down (1 min)
        L4: Walking (1 min)
        L5: Climbing stairs (1 min)
        L6: Waist bends forward (20x)
        L7: Frontal elevation of arms (20x)
        L8: Knees bending (crouching) (20x)
        L9: Cycling (1 min)
        L10: Jogging (1 min)
        L11: Running (1 min)
        L12: Jump front & back (20x)

        :return: data
        '''
        self.name = "UCI mHealth"
        sub_folder = 'UCI/mHealth/'
        sr = 50
        activity_map = {0: 'NULL', 1: 'INACTIVE', 2: 'INACTIVE', 3: 'INACTIVE', 4: 'WALKING', 5: 'STAIRS', 6: 'BEND_FORWARD',
                        7: 'ARM_ELEVATION', 8: 'KNEE_BEND', 9: 'CYCLING', 10: 'JOGGING', 11: 'RUNNING', 12: 'JUMP'}

        # Load the first subject and then the rest iteratively
        data = pd.read_csv(self.root_folder + sub_folder + 'mHealth_subject%d.log' % 1,
                          sep='\t',
                          usecols=[0, 1, 2, 23],
                          names=['x', 'y', 'z', 'labels'])

        data_array = rolling_window(data.values, window=(self.n_samples, 0), step=self.step).swapaxes(1, 2)
        users = np.ones((data_array.shape[0])) * 1
        for idx in range(2, 11):
            tmp = pd.read_csv(self.root_folder + sub_folder + 'mHealth_subject%d.log' % idx,
                                              sep='\t',
                                              usecols=[0, 1, 2, 23],
                                              names=['x', 'y', 'z', 'label']).values
            tmp = rolling_window(tmp, window=(self.n_samples, 0), step=self.step).swapaxes(1, 2)
            data_array = np.concatenate((data_array, tmp))
            users = np.concatenate((users, idx*np.ones(tmp.shape[0])))

        y = np.empty(users.shape)
        for idx in range(users.shape[0]):
            y[idx] = np.argmax(np.bincount(data_array[idx, :, -1].astype('int')))
        data_array = data_array[:, :, :3]

        data_array, stats = self.add_features(data_array,
                                       normalise=self.normalize,
                                       ratio=0,
                                       add_roll=self.add_roll,
                                       add_pitch=self.add_pitch,
                                       add_filter=self.add_filter,
                                       comp_magnitude=self.comp_magnitude)
        y = self.map_to_common_activities(y, activity_map)
        return data_array, y.astype('int'), self.name, users, stats

    def idash(self):
        """
        This dataset contains motion sensor data of 16 physical activities
        (walking, jogging, stair climbing, etc.) collected on 16 adults using
        an iPod touch device (Apple Inc.). The data sampling rate was 30 Hz.
        The collection time for an activity varied from 20 seconds to 17 minutes.
        """
        self.name = "IDASH"
        sub_folder = 'Physical Activity Sensor Data-Public/Public/iDASH_activity_dataset/'
        sr = 30.
        labels = ['400m_brisk_walk_pocket',
                  '400m_jog_pocket',
                  '400m_normal_walk_pocket',
                  '400m_slow_walk_pocket',
                  'Sit_and_walk_test_pocket',
                  'Stair_ascend_brisk_pocket',
                  'Stair_ascend_normal_pocket',
                  'Stair_descend_brisk_pocket',
                  'Stair_descend_normal_pocket',
                  'Step_test_brisk_pocket',
                  'Step_test_normal_pocket',
                  'Treadmill_jog_5.5mph_pocket',
                  'Treadmill_walk_1.5mph_pocket',
                  'Treadmill_walk_3.0mph_pocket',
                  'Treadmill_walk_4.0mph_pocket',
                  'Walking_pocket']
        activity_map = {0: 'WALKING', 1: 'JOGGING', 2: 'WALKING', 3: 'WALKING', 4: 'WALKING', 5: 'STAIRS', 6: 'STAIRS',
                        7: 'STAIRS', 8: 'STAIRS', 9: 'STEP', 10: 'STEP', 11: 'JOGGING', 12: 'WALKING', 13: 'WALKING',
                        14: 'WALKING', 15: 'WALKING'}

        # Load data
        subjects = range(1, 17)
        cols = [0, 1, 2]
        tmp_seg = np.empty((0, self.n_samples * sr/SR, len(cols)))
        y = []
        users = []
        for subject in subjects:
            files = sorted(glob.glob(self.root_folder + sub_folder + '%d/*' % subject))
            for idx, csv_file in enumerate(files):
                if not "blank" in csv_file:
                    tmp = pd.read_csv(csv_file, sep=',', usecols=cols).values
                    tmp = rolling_window(tmp, window=(self.n_samples * sr/SR, 0), step=self.step * sr/SR).swapaxes(1, 2)
                    tmp_seg = np.vstack((tmp_seg, tmp))
                    y.append(np.ones((tmp.shape[0],), dtype=np.int)*idx)
                    users.append(np.ones((tmp.shape[0],), dtype=np.int)*subject)

        users = np.asarray(list(itertools.chain.from_iterable(users)))
        y = list(itertools.chain.from_iterable(y))

        n_windows, sequence_length, n_features = tmp_seg.shape
        data_array = np.empty((n_windows, self.n_samples, n_features))
        for idx in range(n_windows):
            for f in range(n_features):
                data_array[idx, :, f] = resample(tmp_seg[idx, :, f], self.n_samples)

        data_array, stats = self.add_features(data_array,
                                       normalise=self.normalize,
                                       add_roll=self.add_roll,
                                       add_pitch=self.add_pitch,
                                       add_filter=self.add_filter,
                                       comp_magnitude=self.comp_magnitude)
        y = self.map_to_common_activities(y, activity_map)
        return data_array, y.astype('int'), self.name, users, stats

    def nursing(self):
        sub_folder = 'sosolab/Nursing/'
        sr = 50.
        RuntimeError('Not implemented')

    def lingacceleration(self):
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

    def pamap2(self):
        RuntimeError('Not implementet')
        pass

    def mhealth_maninni(self):
        """
        Data from Stanford/MIT. Unfortunately they only provide a comp_magnitude vector
        Subjects: 33
        """
        RuntimeError('Only provides comp_magnitude signal')
        sub_folder = 'mhealth/Mannini_Data_and_Code_PMC2015/Data/'
        file_name = 'StanfordDataset2010_adult_5sensors_win10000_corrected_classes1to3_apr2013_90Hz_filt_o4f20_recalib_PMC.mat'
        data = loadmat(self.root_folder + sub_folder + file_name)
        activities = data['FeatureNames']
        X = data['Data_m']

    def map_to_common_activities(self, y, activity_map):
        return np.asarray([MAP_ACTIVITY.get(activity_map.get(l)) for l in y]).astype('int')

    def add_features(self, data, normalise=True, ratio=0, add_roll=False, add_pitch=False, add_filter=False, comp_magnitude=False):
        n_win, n_samp, n_dim = data.shape
        stats = []

        # data = lowpass_filter(data, fs=50, cutoff=20)

        if ratio > 1:
            data = downsample(data.reshape(-1, n_dim), ratio=ratio).\
                reshape(n_win, n_samp/ratio, n_dim)

        if add_pitch:
            pitches = []
            for i in range(n_win):
                pitches.append(pitch(data[i]))

        if add_roll:
            rolls = []
            for i in range(n_win):
                rolls.append(roll(data[i,:,:n_dim]))

        if comp_magnitude:
            data = magnitude(data).reshape((n_win, n_samp, 1))

        if add_filter:
            enrich_fs = 50
            tmp_lp = split_signal(data, enrich_fs)
            tmp_hp = data - tmp_lp

        # Concat everything
        if add_roll: data = np.concatenate((data, rolls), axis=2)
        if add_pitch: data = np.concatenate((data, pitches), axis=2)
        if add_filter: data = np.concatenate((data, tmp_lp, tmp_hp), axis=2)

        if normalise:
            n_win, n_samp, n_dim = data.shape
            # data_mean = data.reshape((-1, n_dim)).mean(axis=0)
            # data_std = data.reshape((-1, n_dim)).std(axis=0)
            # data = (data.reshape((-1, n_dim)) - data_mean).reshape((n_win, n_samp, n_dim))
            # data = (data.reshape((-1, n_dim)) / data_std).reshape((n_win, n_samp, n_dim))

            for idx in range(data.shape[0]):
                data_mean = data[idx].mean(axis=0)
                data_std = data[idx].std(axis=0)
                data[idx] = data[idx] - data_mean
                data[idx] = data[idx] / data_std

                stats.append([data_mean, data_std])
            stats = np.array(stats)
        return data, stats


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