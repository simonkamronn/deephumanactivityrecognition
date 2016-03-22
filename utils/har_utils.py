import numpy as np
from matplotlib.mlab import specgram
from scipy.signal import butter, lfilter


def roll(data):
    x, y, z = np.transpose(data)
    return np.reshape(np.arctan2(y, z), (len(x), 1))


def pitch(data):
    x, y, z = np.transpose(data)
    return np.reshape(np.arctan2(-x, np.sqrt(y**2 + z**2)), (len(x), 1))


def magnitude(x_in):
    return np.sqrt((x_in*x_in).sum(axis=2))


def spectrogram_3d(data):
    """
    Convert array of epoched accelerometer time series to spectrograms
    :param data: accelerometer data of dim samples x channels x window length
    :return: spectrogram of dim samples x 24 x 24 where channels are concatenated
    """
    data = data[:, :, :3]
    print("Spectrogram conversion input shape: ", data.shape)
    n_bins = 16
    n_win, n_samples, n_fea = data.shape
    nfft = 128
    noverlap = nfft - n_samples/n_bins
    # X = comp_magnitude(np.swapaxes(X[:, 0:3, :], 1, 2).reshape(-1, 3))
    # X = 10. * np.log10(specgram(np.pad(X, pad_width=nfft/2-1, mode='constant'), nfft=nfft, noverlap=noverlap)[0])
    # X = X.reshape(nfft/2+1, 1, n_win, n_samples/(nfft-noverlap)).swapaxes(0, 2)[:, :, :nfft/2]

    ptt = lambda x: 10. * np.log10(specgram(np.pad(x.reshape(-1),
                                                   pad_width=nfft/2-1,
                                                   mode='constant'),
                                            NFFT=nfft,
                                            noverlap=noverlap)[0])
    data = np.rollaxis(data, 2, 0)
    data = np.array([ptt(x) for x in data])
    print("Spectrogram shape", data.shape)
    data = np.reshape(data[:, :n_fea*n_bins], [n_fea, n_fea*n_bins, n_win, n_bins]).swapaxes(0, 2).reshape([n_win, 1, n_fea*n_bins, n_fea*n_bins])
    data = data - np.mean(data)
    print("Spectrogram conversion output shape", data.shape)
    return data


def spectrogram_2d(data):
    """
    Convert array of epoched accelerometer time series to spectrograms
    :param data: accelerometer data of dim samples x window_length
    :return: spectrogram
    """
    print("Spectrogram conversion input shape: ", data.shape)
    n_bins = 16
    n_win, n_samples = data.shape
    nfft = 128
    noverlap = nfft - n_samples/n_bins

    ptt = lambda x: 10. * np.log10(specgram(np.pad(x.reshape(-1),
                                                   pad_width=(noverlap, noverlap),
                                                   mode='edge'),
                                            NFFT=nfft,
                                            noverlap=noverlap)[0])
    data =ptt(data)[:, n_bins:-(n_bins-1)]
    print("Spectrogram shape", data.shape)
    data = np.reshape(data[:nfft/2], [nfft/2, n_win, n_bins]).swapaxes(0, 1).reshape([n_win, 1, nfft/2, n_bins]).swapaxes(2, 3)
    data = data - np.mean(data)
    print("Spectrogram conversion output shape", data.shape)
    return data


# Expand the target to all time steps
def expand_target(y, length):
    return np.rollaxis(np.tile(y, (length, 1, 1)), 1,)


def split_signal(data, fs, cutoff=0.05, order=2):
    n_win, n_samples, n_dim = data.shape
    tmp = np.reshape(data, (n_win*n_samples, n_dim))
    normal_cutoff = cutoff / (0.5 * fs)
    b, a = butter(order, normal_cutoff, 'low', analog=False)
    lp_sig = lfilter(b, a, tmp, axis=0).reshape(n_win, -1, n_dim)

    return lp_sig


def lowpass_filter(data, fs, cutoff=10, order=2):
    n_win, n_samples, n_dim = data.shape
    tmp = np.reshape(data, (n_win*n_samples, n_dim))
    normal_cutoff = cutoff / (0.5 * fs)
    b, a = butter(order, normal_cutoff, 'low', analog=False)
    lp_sig = lfilter(b, a, tmp, axis=0).reshape(n_win, -1, n_dim)

    return lp_sig


def wavelet_decomp(data, level=3):
    pass


def rolling_window_lastaxis(a, window, step):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError, "`window` must be at least 1."
    if window > a.shape[-1]:
        raise ValueError, "`window` is too long: %d" % window
    shape = a.shape[:-1] + ((a.shape[-1] - window + step)/step, window)
    strides = a.strides[:-1] + (a.strides[-1]*step,) + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a, window, step=1):
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window, step)
    for i, win in enumerate(window):
        if 1 < win < a.shape[0]:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win, step)
            a = a.swapaxes(-2, i)
    return a


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


def window_segment(data, window=64):
    # Segment in windows on axis 1
    n_samp, n_dim = data.shape
    n_win = n_samp//(window)
    data = np.reshape(data[:n_win * window], (n_win, window, n_dim))
    return data