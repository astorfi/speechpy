from __future__ import division
import numpy as np
import processing
from scipy.fftpack import dct
import math
from functions import *


def filterbanks(num_filter, fftpoints, sampling_freq, low_freq=None, high_freq=None):
    """Compute the Mel-filterbanks. Each filter will be stored in one rows.
                                    The columns correspond to fft bins.

    :param num_filter: the number of filters in the filterbank, default 20.
    :param fftpoints: the FFT size. Default is 512.
    :param sampling_freq: the samplerate of the signal we are working with. Affects mel spacing.
    :param low_freq: lowest band edge of mel filters, default 0 Hz
    :param high_freq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size num_filter x (fftpoints//2 + 1) which are filterbank
    """
    high_freq = high_freq or sampling_freq / 2
    low_freq = low_freq or 300
    assert high_freq <= sampling_freq / 2, "High frequency cannot be greater than half of the sampling frequency!"
    assert low_freq >= 0, "low frequency cannot be less than zero!"

    ######################################################
    ########### Computing the Mel filterbank #############
    ######################################################

    # converting the upper and lower frequencies to Mels.
    # num_filter + 2 is because for num_filter filterbanks we need num_filter+2 point.
    mels = np.linspace(frequency_to_mel(low_freq), frequency_to_mel(high_freq), num_filter + 2)

    # we should convert Mels back to Hertz because the start and end-points should be at the desired frequencies.
    hertz = mel_to_frequency(mels)

    # The frequency resolution required to put filters at the
    # exact points calculated above should be extracted.
    #  So we should round those frequencies to the closest FFT bin.
    freq_index = (np.floor((fftpoints + 1) * hertz / sampling_freq)).astype(int)

    # Initial definition
    filterbank = np.zeros([num_filter, fftpoints])

    # The triangular function for each filter
    for i in range(0, num_filter):
        left = int(freq_index[i])
        middle = int(freq_index[i + 1])
        right = int(freq_index[i + 2])
        z = np.linspace(left, right, num=right - left + 1)
        filterbank[i, left:right + 1] = triangle(z, left=left, middle=middle, right=right)

    return filterbank

def mfcc(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,num_cepstral =13,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, dc_elimination=True):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N x 1 array
    :param sampling_frequency: the sampling frequency of the signal we are working with.
    :param frame_length: the length of each frame in seconds. Default is 0.020s
    :param frame_stride: the step between successive frames in seconds. Default is 0.02s (means no overlap)
    :param num_filters: the number of filters in the filterbank, default 40.
    :param fft_length: number of FFT points. Default is 512.
    :param low_frequency: lowest band edge of mel filters. In Hz, default is 0.
    :param high_frequency: highest band edge of mel filters. In Hz, default is samplerate/2
    :param num_cepstral: Number of cepstral coefficients.
    :param dc_elimination: hIf the first dc component should be eliminated or not.

    :returns: A numpy array of size (num_frames x num_cepstral) containing mfcc features.
    """

    feature, energy = mfe(signal, sampling_frequency=sampling_frequency, frame_length=frame_length, frame_stride=frame_stride,
             num_filters=num_filters, fft_length=fft_length, low_frequency=low_frequency, high_frequency=high_frequency)
    feature = np.log(feature)
    feature = dct(feature, type=2, axis=-1, norm='ortho')[:, :num_cepstral]

    # replace first cepstral coefficient with log of frame energy for DC elimination.
    if dc_elimination:
        feature[:, 0] = np.log(energy)
    return feature


def mfe(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,
          num_filters=40, fft_length=512, low_frequency=0, high_frequency=None):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N x 1 array
    :param sampling_frequency: the sampling frequency of the signal we are working with.
    :param frame_length: the length of each frame in seconds. Default is 0.020s
    :param frame_stride: the step between successive frames in seconds. Default is 0.02s (means no overlap)
    :param num_filters: the number of filters in the filterbank, default 40.
    :param fft_length: number of FFT points. Default is 512.
    :param low_frequency: lowest band edge of mel filters. In Hz, default is 0.
    :param high_frequency: highest band edge of mel filters. In Hz, default is samplerate/2

    :returns:
              features: the energy of fiterbank: num_frames x num_filters
              frame_energies: the energy of each frame: num_frames x 1
    """

    # Convert to float
    signal = signal.astype(float)

    # Stack frames
    frames = processing.stack_frames(signal, sampling_frequency=sampling_frequency, frame_length=frame_length,
                                     frame_stride=frame_stride,
                                     Filter=lambda x: np.ones((x,)),
                                     zero_padding=False)

    # getting the high frequency
    high_frequency = high_frequency or sampling_frequency / 2

    # calculation of the power sprectum
    power_spectrum = processing.power_spectrum(frames, fft_length)
    number_fft_coefficients = power_spectrum.shape[1]
    frame_energies = np.sum(power_spectrum, 1)  # this stores the total energy in each frame

    # Handling zero enegies.
    frame_energies = zero_handling(frame_energies)

    # Extracting the filterbank
    filter_banks = filterbanks(num_filters, number_fft_coefficients, sampling_frequency, low_frequency, high_frequency)

    # Filterbank energies
    features = np.dot(power_spectrum, filter_banks.T)
    features = zero_handling(features)

    return features, frame_energies


def lmfe(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N x 1 array
    :param sampling_frequency: the sampling frequency of the signal we are working with.
    :param frame_length: the length of each frame in seconds. Default is 0.020s
    :param frame_stride: the step between successive frames in seconds. Default is 0.02s (means no overlap)
    :param num_filters: the number of filters in the filterbank, default 40.
    :param fft_length: number of FFT points. Default is 512.
    :param low_frequency: lowest band edge of mel filters. In Hz, default is 0.
    :param high_frequency: highest band edge of mel filters. In Hz, default is samplerate/2

    :returns:
              features: the energy of fiterbank: num_frames x num_filters
              frame_log_energies: the log energy of each frame: num_frames x 1
    """

    feature, frame_energies = mfe(signal, sampling_frequency=sampling_frequency, frame_length=frame_length,
                                 frame_stride=frame_stride,
                                 num_filters=num_filters, fft_length=fft_length, low_frequency=low_frequency,
                                 high_frequency=high_frequency)
    feature = np.log(feature)


    return feature

def extract_derivative_feature(feature):
    """
    This function extracts temporal derivative features which are first and second derivatives.
    :param feature: The feature vector which its size is: N x M

    :return: The feature cube vector which contains the static, first and second derivative features
             size: N x M x 3
    """
    first_derivative_feature = processing.Derivative_Feature_Fn(feature, DeltaWindows=2)
    second_derivative_feature = processing.Derivative_Feature_Fn(first_derivative_feature, DeltaWindows=2)

    # Creating the future cube for each file
    feature_cube = np.concatenate(
        (feature[:, :, None], first_derivative_feature[:, :, None],
         second_derivative_feature[:, :, None]),
        axis=2)
    return feature_cube
