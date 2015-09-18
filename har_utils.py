__author__ = 'Simon'
import numpy as np
from matplotlib.mlab import specgram

def roll(data):
    x, y, z = np.transpose(data)
    return np.arctan(y/np.sqrt(x**2 + z**2)).reshape((len(x), 1))


def pitch(data):
    x, y, z = np.transpose(data)
    return np.arctan(x/np.sqrt(y**2 + z**2)).reshape((len(x), 1))


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
    # X = magnitude(np.swapaxes(X[:, 0:3, :], 1, 2).reshape(-1, 3))
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