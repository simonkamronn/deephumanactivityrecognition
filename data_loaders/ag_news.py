import os
import shutil
import numpy as np
from utils import env_paths
import urllib
import tarfile
import pandas as pd
from data_helper import create_semi_supervised, pad_targets

def _load(max_parse_size):
    """
    Download the AG News dataset if it is not present.
    :return: The train, test and validation set.
    """

    source_path = env_paths.get_data_path("ag_news")
    package_path = os.path.join(source_path, "ag_news_csv.tar.gz")
    extracted_path = os.path.join(source_path, "ag_news_csv")
    processed_path = os.path.join(source_path, "dataset_%i.npz" % max_parse_size)

    if os.path.isfile(processed_path):
        train_set, test_set = np.load(open(processed_path, "rb"))
        return train_set, test_set, None

    origin = (
        "https://drive.google.com/uc?id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms&export=download"
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, package_path)

    def csv_files(members):
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".csv":
                yield tarinfo

    print 'Extracting data to %s' % extracted_path
    tar = tarfile.open(package_path)
    tar.extractall(path=source_path, members=csv_files(tar))
    tar.close()

    train = pd.read_csv(os.path.join(extracted_path, "train.csv"), header=None).values
    test = pd.read_csv(os.path.join(extracted_path, "test.csv"), header=None).values

    # remove downloaded and extracted files.
    os.remove(package_path)
    shutil.rmtree(extracted_path)

    def transform(dat):
        dat[:, 1] += " " + dat[:, -1]
        dat = np.delete(dat, -1, axis=1)
        x = dat[:, 1]
        t = np.array(dat[:, 0], dtype='float32')
        return x, t

    print 'Transforming data'
    train_set = transform(train)
    test_set = transform(test)

    def _parse(xy, max_size):
        alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\|_@#$%^&*~`+-=<>()[]{}"))
        x, y = xy
        new_x = np.zeros((x.shape[0], max_size))
        for i in range(x.shape[0]):
            line = list(x[i].lower())
            for j in range(len(line[:max_size])):
                char = line[j]
                if char in alphabet:
                    idx = np.where(alphabet == char)[0][0]
                    new_x[i, j] = idx
        y -= 1.
        return new_x, y

    print 'Parsing data'
    train_set = _parse(train_set, max_parse_size)
    test_set = _parse(test_set, max_parse_size)

    print 'Dump data'
    np.save(open(processed_path, "wb"), (train_set, test_set))

    return train_set, test_set, None


def load_supervised(max_parse_size=1014):
    """
    Load the AG News dataset.
    :param max_parse_size: The fixed length of a string. The largest needed is 1012.
    :return: The train, test and validation sets.
    """
    train_set, test_set, valid_set = _load(max_parse_size)

    test_set = pad_targets(test_set)
    valid_set = pad_targets(valid_set)
    train_set = pad_targets(train_set)

    return train_set, test_set, valid_set


def load_semi_supervised(n_labeled=100, max_parse_size=1012, seed=123456):
    """
    Load the AG News dataset where only a fraction of data points are labeled. The amount
    of labeled data will be evenly distributed accross classes.
    :param n_labeled: Number of labeled data points.
    :param max_parse_size: The fixed length of a string. The largest needed is 1012.
    :param seed: The seed for the pseudo random shuffle of data points.
    :return: Train set unlabeled and labeled, test set, validation set.
    """

    train_set, test_set, valid_set = _load(max_parse_size)

    rng = np.random.RandomState(seed=seed)
    n_classes = train_set[1].max() + 1
    print n_classes

    # Create the labeled and unlabeled data evenly distributed across classes.
    x_l, y_l, x_u, y_u = create_semi_supervised(train_set, n_labeled, rng)

    train_set = (x_u, y_u)
    train_set_labeled = (x_l, y_l)

    # shuffle data
    train_x, train_t = train_set
    train_collect = np.append(train_x, train_t, axis=1)
    rng.shuffle(train_collect)
    train_set = (train_collect[:, :-n_classes], train_collect[:, -n_classes:])

    test_set = pad_targets(test_set)
    if valid_set is not None:
        valid_set = pad_targets(valid_set)

    return train_set, train_set_labeled, test_set, valid_set

train_set, train_set_u, test_set, _ = load_semi_supervised()
print "done"