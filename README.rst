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

====================	===========
Parameter 		Description
====================	===========
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
====================	===========


Filterbank Features
===================

The attributes for ``filterbank energies`` are the same for ``log_filterbank energies`` too.

===================	===========
Parameter 		Description
===================	===========
signal                  the audio signal from which to compute features. Should be an N x 1 array
sampling_frequency      the sampling frequency of the signal we are working with.
frame_length            the length of each frame in seconds. Default is 0.020s
frame_stride            the step between successive frames in seconds. Default is 0.02s (means no overlap)
num_filters             the number of filters in the filterbank, default 40.
fft_length              number of FFT points. Default is 512.
low_frequency           lowest band edge of mel filters. In Hz, default is 0.
high_frequency          highest band edge of mel filters. In Hz, default is samplerate/2
====================	===========

