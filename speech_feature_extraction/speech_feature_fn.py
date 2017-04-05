import scipy.io as sio
import numpy as np
import os
import sys
from main import fbank
from main import logfbank
from main import mfcc

def ExtractFeature(signal,fs, feature_type):
    """
    This function get the stacked frames of signal and call the relevant functions for extracting features.

    Arg:
        signal: The stacked frames of signal.
        fs: The frequency of the signal.
        feature_type: The desired features.

    output:

    The extracted features.
    """

    if feature_type == 'fbank_energy':
        out_signal = fbank(signal, samplerate=fs)

    elif feature_type == 'logfbank_energy':
        out_signal = logfbank(signal, samplerate=fs)

    elif feature_type == 'MFCC':
        out_signal = mfcc(signal, samplerate=fs)

    else:
        out_signal = signal

    return out_signal