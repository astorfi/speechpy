======================
speech_feature_extraction 
======================

This library provides most frquent used speech features including MFCCs and filterbank energies alogside wi logenergy of filterbanks.
If you are not sure what MFCCs are, and would like to know more have a look at this nice 
`MFCC tutorial <http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/>`_

Installation
============

To install From this repository::

	git clone https://github.com/astorfi/speech_feature_extraction.git
	python setup.py develop


Supported Features
=====
- Mel Frequency Cepstral Coefficients(MFCCs)
- Filterbank Energies
- Log Filterbank Energies(MFECs!)

MFCC Features
=============

The supported attributes for generating MFCC features:

=============	===========
Parameter 		Description
=============	===========
signal			the audio signal from which to compute features. Should be an N*1 array
sampling_frequency	samplerate of the signal we are working with
winlen			the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
winstep			the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)
nfilt			the number of filters in the filterbank, default 26.
nfft			the FFT size. Default is 512.
lowfreq			lowest band edge of mel filters. In Hz, default is 0
highfreq		highest band edge of mel filters. In Hz, default is samplerate/2
preemph			apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97
returns			A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The second return value is the energy in each frame (total energy, unwindowed)
=============	===========


=============   ===========
Parameter 		Description
=============   ===========
signal                  the audio signal from which to compute features. Should be an N x 1 array
sampling_frequency      the sampling frequency of the signal we are working with.
frame_length            the length of each frame in seconds. Default is 0.020s
frame_stride            the step between successive frames in seconds. Default is 0.02s (means no overlap)
num_filters             the number of filters in the filterbank, default 40.
fft_length              number of FFT points. Default is 512.
low_frequency           lowest band edge of mel filters. In Hz, default is 0.
high_frequency          highest band edge of mel filters. In Hz, default is samplerate/2
num_cepstral            number of cepstral coefficients.
dc_elimination          if the first dc component should be eliminated or not.
=============   ===========


Filterbank Features
===================

These filters are raw filterbank energies. 
For most applications you will want the logarithm of these features.
The default parameters should work fairly well for most cases. 
If you want to change the fbank parameters, the following parameters are supported::

	python
	def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)

=============	===========
Parameter 		Description
=============	===========
signal			the audio signal from which to compute features. Should be an N*1 array
samplerate		the samplerate of the signal we are working with
winlen			the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
winstep			the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)
nfilt			the number of filters in the filterbank, default 26.
nfft			the FFT size. Default is 512.
lowfreq			lowest band edge of mel filters. In Hz, default is 0
highfreq		highest band edge of mel filters. In Hz, default is samplerate/2
preemph			apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97
returns			A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The second return value is the energy in each frame (total energy, unwindowed)
=============	===========


Reference
=========
sample english.wav obtained from::

	wget http://voyager.jpl.nasa.gov/spacecraft/audio/english.au
	sox english.au -e signed-integer english.wav
