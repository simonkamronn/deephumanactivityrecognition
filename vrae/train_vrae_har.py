#!/usr/bin/env python

import VRAE
import numpy as np
import pylab as pl
import theano
theano.config.floatX = 'float64'
from data_preparation.load_data import LoadHAR

def run():
    n_samples, step = 200, 200
    load_data = LoadHAR(add_pitch=True, add_roll=True, add_filter=True, n_samples=n_samples,
                        step=step, normalize=True, comp_magnitude=False)
    X, y, name, users, stats = load_data.uci_hapt()
    X = np.concatenate((X, stats), axis=1)
    features = X.shape[-1]
    data = np.reshape(X[:100], (1, -1, features)).swapaxes(1, 2).astype(np.float32)
    print('Data shape, ', data.shape)
    # wavlen = wavdata.shape[1]
    # chunklen = 1
    # numchunks = wavlen/chunklen
    # data = wavdata[0,:numchunks*chunklen].reshape((1, chunklen, numchunks))

    hidden_units_encoder = 100
    hidden_units_decoder = hidden_units_encoder
    latent_variables = 10
    b1 = 0.05
    b2 = 0.001
    learning_rate = 1e-3
    sigma_init = 1e-3
    batch_size = 100

    vrae = VRAE.VRAE(hidden_units_encoder, hidden_units_decoder, features, latent_variables, b1, b2, learning_rate, sigma_init, batch_size)

    print("create_gradientfunctions")
    tdata = theano.shared(data.astype(np.float32), borrow=True)
    vrae.create_gradientfunctions(tdata)

    print("save_parameters")
    # vrae.save_parameters("data/")

    print("plotting input")
    # print(data[0])

    N = 100
    xs = np.zeros((N, 1))
    xss = []
    zs = []

    for i in range(N):
        print("encoding")
        # z, mu_encoder, log_sigma_encoder = vrae.encode(data[0,:1].T)
        z, mu_encoder, log_sigma_encoder = vrae.encode(data[0].T[(i*100):(i+1)*100])
        zs.append(z)

        print("z.shape, z, mu_enc, s_enc", z.shape, mu_encoder, log_sigma_encoder)
        # np.save("z.npy", z)

        print("decoding")
        tx = vrae.decode(10, latent_variables, z)
        xs[i, 0] = tx[-1]
        xss.append(tx)
        # print("x.shape, x", x.shape, x)

    zs = np.asarray(zs)
    print(zs.shape)
    xss = np.asarray(xss)
    print(xss.shape)

    pl.subplot(311)
    pl.plot(data[0].T[:1000])
    pl.subplot(312)
    pl.plot(zs[:,:,0].T)
    pl.subplot(313)
    pl.plot(xs)
    pl.plot(xss[:,:,0].T)
    pl.show()

# wavfile.write("x.wav", 44100, x)


if __name__ == '__main__':
    run()
