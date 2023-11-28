import librosa
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.decomposition import PCA
from python_speech_features import mfcc

np.set_printoptions(precision=32, suppress=True)


def psd(data):
    # f contains the frequency components
    # S is the PSD
    f, S = periodogram(data, 100.0, scaling='density')
    f = [f[i] for i in range(len(f))]
    S = [float(round(S[i], 32)*10**10) for i in range(len(S))] # 데이터 표현을 위해 소수점 32 자리에서 반올림 + 10^10 을 곱해줌
    # print(type(S))
    # print(type(S[0]))
    return f[:26], S[:26]


# Frequency-Domain Analysis
def fft(data, fixSize=40):
    # print(f'length of data {len(data)}')
    fft = np.fft.fft(data) / len(data)
    fft_magnitude = abs(fft)
    fft_magnitude = [float(fft_magnitude[i]) for i in range(len(fft_magnitude))]
    # print(len(fft_magnitude))
    fft_magnitude = [i for i in fft_magnitude]
    return fft_magnitude[:fixSize]


def acf(data, k):
    data = pd.Series(data)
    mean = data.mean()
    denominator = np.sum(np.square(data - mean))
    numerator = np.sum((data - mean) * (data.shift(k) - mean))
    acf_val = numerator / denominator
    return acf_val


def mean_std(data):
    return data.mean(), data.std()


def min_max(data):
    return data.min(), data.max()


# Time-Domain Analysis
def zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum()


def signal_magnitude_area(data):
    return np.sum(np.abs(data))


# Temporal Features
def consecutive_diff(data):
    return np.diff(data)


# Advanced Techniques
def pca_features(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def compute_mfcc(audio_data, sample_rate):
    #audio_data = audio_data - np.mean(audio_data)
    #audio_data = audio_data / np.max(audio_data)
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.010, winstep=0.01,
                     numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)

    mfccSqueezed = mfcc_feat.T.reshape(1, -1)[0]
    mfccSqueezed = [i for i in mfccSqueezed]

    return mfccSqueezed