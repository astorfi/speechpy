import scipy.io.wavfile as wav
import numpy as np
import os
import sys
from speechpy import processing
from speechpy import feature

# content of test_class.py
# content of test_class.py
class TestAttributes(object):
    def test_processing(self):
        
        # Cheching the availibility of functions in the chosen attribute
        assert hasattr(processing, 'preemphasis')
        assert hasattr(processing, 'stack_frames')
        assert hasattr(processing, 'fft_spectrum')
        assert hasattr(processing, 'power_spectrum')
        assert hasattr(processing, 'log_power_spectrum')
        assert hasattr(processing, 'derivative_extraction')
        assert hasattr(processing, 'cmvn')
        assert hasattr(processing, 'cmvnw')

    def test_feature(self):
        
        # Cheching the availibility of functions in the chosen attribute
        assert hasattr(feature, 'filterbanks')
        assert hasattr(feature, 'mfcc')
        assert hasattr(feature, 'mfe')
        assert hasattr(feature, 'lmfe')
        assert hasattr(feature, 'extract_derivative_feature')
