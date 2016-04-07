import scipy

def stft(x, framesz, hop):
    w = scipy.hamming(framesz)
    X = scipy.array([scipy.fft(w*x[i:i+framesz]) for i in range(0, len(x)-framesz, hop)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x