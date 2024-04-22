import librosa
import numpy as np
from scipy import signal

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load('rock/rock_0.mp3', mono=False)

D_stereo = librosa.stft(y)
S_stereo = np.abs(D_stereo)

# Get the default Fourier frequencies
freqs = librosa.fft_frequencies(sr=sr)

# We'll interpolate the first five harmonics of each frequency
harmonics = [1, 2, 3, 4, 5]

S_harmonics = librosa.interp_harmonics(S_stereo, freqs=freqs, h_range=harmonics)