import scipy.io.wavfile as wav
import numpy as np
from main import energy_feature
from main import logenergy_feature
from main import mfcc_feature
from main import extract_derivative_feature

file_name = 'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav'
fs, signal = wav.read(file_name)
signal = signal[:,0]
out = mfcc_feature(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
out = extract_derivative_feature(out)

