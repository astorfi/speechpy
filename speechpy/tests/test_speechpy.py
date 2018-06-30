import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('../..'))
print(lib_path)
sys.path.append(lib_path)
from speechpy import processing
from speechpy import feature
from speechpy import functions

# Ramdom signal generation for testing
mu, sigma = 0, 0.1 # mean and standard deviation
signal = np.random.normal(mu, sigma, 1000000)
fs = 16000

 # Generating stached frames with SpeechPy
frame_length = 0.02
frame_stride = 0.02
frames = processing.stack_frames(signal, sampling_frequency=fs,
                                  frame_length=frame_length,
                                  frame_stride=frame_stride,
                                  filter=lambda x: np.ones((x,)),
                                  zero_padding=True)

class Test_Methods_Exists(object):
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
        
    def test_functions(self):
    
        # Cheching the availibility of functions in the chosen attribute
        assert hasattr(functions, 'frequency_to_mel')
        assert hasattr(functions, 'mel_to_frequency')
        assert hasattr(functions, 'triangle')
        assert hasattr(functions, 'zero_handling')
        

class Test_Processing(object):

    def test_preemphasis(self):
       
       # Performing the operation on the generated signal.
       signal_preemphasized = processing.preemphasis(signal, cof=0.98)
       
       # Shape matcher
       assert signal_preemphasized.ndim == 1
       assert signal_preemphasized.shape == signal.shape
       
    def test_stack_frames(self):
                
        # Direct calculation using numpy
        window = int(np.round(frame_length * fs))
        step = int(np.round(frame_stride * fs))
        all_frames = (int(np.ceil((signal.shape[0]
                                      - window) / step)))
        
        # Shape matching of stacked frames
        assert all_frames == frames.shape[0]
    
    def test_cmvn(self):
        
        normalized_feature = processing.cmvn(frames, variance_normalization=True)
        
        # Shape match
        assert normalized_feature.shape == frames.shape
        
        # Check the std and mean of the output vector
        assert np.allclose(np.mean(normalized_feature,axis=0), np.zeros((1,normalized_feature.shape[1])))
        assert np.allclose(np.std(normalized_feature,axis=0), np.ones((1,normalized_feature.shape[1])))
        
        

        
        
        
        
        
        
