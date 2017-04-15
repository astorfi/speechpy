import scipy.io.wavfile as wav
import numpy as np
import speechpy

file_name = 'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav'
fs, signal = wav.read(file_name)
signal = signal[:,0]

signal = speechpy.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.020, Filter=lambda x: np.ones((x,)),
         zero_padding=True)

print(signal.shape)
sys.exit(1)

############# Extract MFCC features #############
mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
mfcc_feature_cube = speechpy.extract_derivative_feature(mfcc)
print('mfcc feature cube shape=', mfcc_feature_cube.shape)

############# Extract logenergy features #############
logenergy = speechpy.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
logenergy_feature_cube = speechpy.extract_derivative_feature(logenergy)
print('logenergy features=', logenergy.shape)



