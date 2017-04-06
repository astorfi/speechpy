======================
speech_feature_extraction 
======================

This library provides most frquent used speech features including MFCCs and filterbank energies alogside wi logenergy of filterbanks.
If you are not sure what MFCCs are, and would like to know more have a look at this nice 
`MFCC tutorial <http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/>`_

Installation
============

To install From this repository at first the repository must be cloned::

	git clone https://github.com/astorfi/speech_feature_extraction.git
	
After cloning the reposity, root to the repository directory then execute::	
	
	python setup.py develop


Supported Features
=====
- Mel Frequency Cepstral Coefficients(MFCCs)
- Filterbank Energies
- Log Filterbank Energies(MFECs!)

MFCC Features
=============

The supported attributes for generating MFCC features can be seen by investigating the related function:

.. code-block:: python
      
      def mfcc_feature(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,num_cepstral =13,
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


Filterbank Energy Features
==========================

The attributes for ``filterbank energies`` are the same for ``log_filterbank energies`` too.

.. code-block:: python

	def energy_feature(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,
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
	   
Test Example
==========================

The test example can be seen in ``test/test.py`` as below:

.. code-block:: python

	import scipy.io.wavfile as wav
	import numpy as np
	import speech_feature_extraction
        
	# Read the sample audio file
	file_name = 'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav'
	fs, signal = wav.read(file_name)
	
	# Signal shape must be (N,)
	signal = signal[:,0]

	############# Extract MFCC features #############
	mfcc = speech_feature_extraction.mfcc_feature(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
		     num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
	mfcc_feature_cube = speech_feature_extraction.extract_derivative_feature(mfcc)
	print('mfcc feature cube shape=', mfcc_feature_cube.shape)

	############# Extract logenergy features #############
	logenergy = speech_feature_extraction.logenergy_feature(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
		     num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
	logenergy_feature_cube = speech_feature_extraction.extract_derivative_feature(logenergy)
	print('logenergy feature cube shape=', logenergy.shape)
	
For ectracting the feature at first, the signal samples will be stacked into frames. The features are computed for each frame in the stacked frames collection.

Dependencies
==========================

Two packages of ``Scipy`` and ``NumPy`` are the required dependencies which will be installed automatically by running the ``setup.py`` file.
