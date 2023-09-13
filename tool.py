from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# %matplotlib inline
from scipy import signal
from scipy.io import wavfile 
import IPython.display as ipd
import librosa
import librosa.display 
from matplotlib import cm
import os

np.random.seed(666)
title = ('')

def _fourier(wave, sr, ax=None, title=None, f_ratio=0.1):
    X = np.fft.fft(wave)
    X_mag = np.absolute(X)
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag)*f_ratio)  
    plt.figure(figsize=(25, 10))
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('time (s)')
    plt.ylabel('frequencies (Hz)')
    plt.title(title)

def _spectrogram(wave, sr, frame_size=1024, hop_size=512, y_axis="linear"):
    wave_stft = librosa.stft(wave, n_fft=frame_size, hop_length=hop_size)
    wave_stft_abs = np.abs(wave_stft) ** 2
    plt.xlabel('time (s)')
    plt.ylabel('frequencies (Hz)')
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(wave_stft_abs, sr=sr, hop_length=hop_size, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    
    
def _spectrogram_3d(wave, sr, frame_size=1024, hop_size=512, y_axis="linear"):
    wave_stft = librosa.stft(wave, n_fft=frame_size, hop_length=hop_size)
    wave_stft_abs = np.abs(wave_stft) ** 2
    
    X, Y, Z = t[None, :], freqs[:, None],  20.0 * np.log10(spec)
    
    
    plt.axes(projection='3d')
    plt.xlabel('time (s)')
    plt.ylabel('frequencies (Hz)')
    plt.figure(figsize=(25, 10))
    plt.plot_surface(X, Y, Z, cmap='binary')
#     librosa.display.specshow(wave_stft_abs, sr=sr, hop_length=hop_size, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    
def _specgram2d(y, srate=44100, ax=None, title=None):
    fs = 10e4
    if not ax:
        ax = plt.axes()
    ax.set_title(title, loc='center', wrap=True)
    spec, freqs, t, im = ax.specgram(y, Fs=fs, scale='dB', vmax=0, cmap='binary')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequencies (Hz)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude (dB)')
    cbar.minorticks_on()
    return spec, freqs, t, im

def _specgram3d(y, srate=44100, ax=None, title=None):
    fs = 10e3
    if not ax:
        ax = plt.axes(projection='3d')
    ax.set_title(title, loc='center', wrap=True)
    
    spec, freqs, t = mlab.specgram(y, Fs=srate)
    
    X, Y, Z = t[None, :], freqs[:, None],  20.0 * np.log10(spec)
    ax.plot_surface(X, Y, Z, cmap='binary')
#     ax.plot_surface(Y, X, Z, cmap='viridis')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequencies (Hz)')
    ax.set_zlabel('amplitude (dB)')
    
#     ax.set_ylim(0, 16000)
    
    ax.view_init(elev=50, azim=45)
    return X, Y, Z



