"""
This example is provided to test the package locally.
There is no need to installing the package using pip. 
Only forking the project repository is required.
"""

import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('../..'))
print(lib_path)
sys.path.append(lib_path)
from speechpy import processing
from speechpy import feature
import os

file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav')
fs, signal = wav.read(file_name)
signal = signal[:,0]

# Pre-emphasizing.
signal_preemphasized = processing.preemphasis(signal, cof=0.98)

# Staching frames
frames = processing.stack_frames(signal, sampling_frequency=fs,
                                          frame_length=0.020,
                                          frame_stride=0.01,
                                          filter=lambda x: np.ones((x,)),
                                          zero_padding=True)

# Extracting power spectrum
power_spectrum = processing.power_spectrum(frames, fft_points=512)
print('power spectrum shape=', power_spectrum.shape)

############# Extract MFCC features #############
mfcc = feature.mfcc(signal, sampling_frequency=fs,
                             frame_length=0.020, frame_stride=0.01,
                             num_filters=40, fft_length=512, low_frequency=0,
                             high_frequency=None)
mfcc_cmvn = processing.cmvnw(mfcc,win_size=301,
                                      variance_normalization=True)
print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

# Extracting derivative features
mfcc_feature_cube = feature.extract_derivative_feature(mfcc)
print('mfcc feature cube shape=', mfcc_feature_cube.shape)

############# Extract logenergy features #############
logenergy = feature.lmfe(signal, sampling_frequency=fs,
                                  frame_length=0.020, frame_stride=0.01,
                                  num_filters=40, fft_length=512,
                                  low_frequency=0, high_frequency=None)
logenergy_feature_cube = feature.extract_derivative_feature(logenergy)
print('logenergy features=', logenergy.shape)





