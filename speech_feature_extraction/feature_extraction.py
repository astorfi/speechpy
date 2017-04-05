import numpy
import glob
import os
import sys
import numpy as np
from speech_feature_fn import ExtractFeature
import processing
from main import fbank
from main import logfbank
from main import mfcc


def stack_signal(signal, fs, frame_length, overlap_factor):
    """

    signal: The input signal which is extracted from a sound file.
    fs: The frequency of the signal.
    frame_length: The length of the frame from which the energy coefficients will be extracted.
    overlap_factor: The overlapping factor for frames. default=0 which means no overlapping.

    :return: The stacked signal
    """

    # Convert to float
    signal = signal.astype(float)

    # Stack frames
    stacked_signal = processing.Stack_Frames(signal, fs, frame_length, overlap_factor,
                                             Filter=lambda x: np.ones((x,)),
                                             zero_padding=True)

    return stacked_signal


def feature_extraction(stacked_signal, fs, feature_type='logfbank_energy'):
    """
    This function get the ids and associated folders for generating features.
    The output is as numpy arrays.

    Arg:

        choice_stack: If the frames needs to be stack or not.

    output:

    procedure:

    This file will generate cube of features. The procedure is as follows:

        1- Stacking frames will be done using "processing.Stack_Frames".

        2- For each stacked sequence of frames, the feature vectors(static, first and second order
        derivatives) will be extracted with shape: (frames,feature_size). The feature_size will
        be called Mels because it is associated with the number of chosen mel-frequencies.

        3- Using the three aforementioned features, the feature cube will be generated with
        the (frames, spectral_features ,Channels) dimensions.

        5- The final cube shape is (frames, spectral_features ,Channels) == (channel,height,width).
        * Mels is the number of filterbanks/specific frequencies.
        * Channels fill like the following:
        channel[0]: static features
        channel[1]: first order derivatives
        channel[2]: second order derivatives

    """

    # Feature extraction.
    static_feature = ExtractFeature(stacked_signal, fs, feature_type)
    first_derivative_feature = processing.Derivative_Feature_Fn(static_feature, DeltaWindows=2)
    second_derivative_feature = processing.Derivative_Feature_Fn(first_derivative_feature, DeltaWindows=2)

    # Creating the future cube for each file
    feature = np.concatenate(
        (static_feature[:, :, None], first_derivative_feature[:, :, None],
         second_derivative_feature[:, :, None]),
        axis=2)

    return feature
