from utils import env_paths
import tables
import os


def load():
    data_file = 'har_data.h5'
    new_path = os.path.join(
        env_paths.get_data_path("har"),
        data_file
    )

    if not os.path.isfile(new_path):
        ValueError('No file: %s' % new_path)

    print("Loading data from %s" % new_path)
    hdf5_file = tables.open_file(new_path, mode='r', title='HAR data')
    X = hdf5_file.root.X[:].astype('float32')
    y = hdf5_file.root.y[:].astype('int')
    users = hdf5_file.root.users[:].astype('int')
    stats = hdf5_file.root.stats[:].astype('float32')
    hdf5_file.close()

    return X, y, users, stats