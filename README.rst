==========================
speech_feature_extraction 
==========================

.. image:: https://circleci.com/gh/astorfi/speech_feature_extraction.svg?style=svg
    :target: https://circleci.com/gh/astorfi/speech_feature_extraction
.. image:: https://travis-ci.org/astorfi/speech_feature_extraction.svg?branch=master
    :target: https://travis-ci.org/astorfi/speech_feature_extraction
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/astorfi/speech_feature_extraction/issues
.. image:: https://coveralls.io/repos/github/astorfi/speech_feature_extraction/badge.svg?branch=master
    :target: https://coveralls.io/github/astorfi/speech_feature_extraction?branch=master
.. image:: https://badge.fury.io/py/speechpy.svg
    :target: https://badge.fury.io/py/speechpy





This library provides most frquent used speech features including MFCCs and filterbank energies alogside with the logenergy of filterbanks.
If you are interested to see what are MFCCs and how they are generated please refer to this 
`wiki <https://github.com/astorfi/speech_feature_extraction/wiki/>`_ page.

.. image:: https://github.com/astorfi/speech_feature_extraction/blob/master/images/speech.gif 

How to Install?
===============

There are two possible ways for installation of this package: local installation and PyPi.

Local Installation
~~~~~~~~~~~~~~~~~~~

For local installation at first the repository must be cloned::

	git clone https://github.com/astorfi/speech_feature_extraction.git
	
After cloning the reposity, root to the repository directory then execute::	
	
	python setup.py develop

Pypi
~~~~~~~~

The package is available on PyPi. For direct installation simply execute the following:

.. code-block:: shell
     
     pip install speechpy


What Features are supported?
=============================
- Mel Frequency Cepstral Coefficients(MFCCs)
- Filterbank Energies
- Log Filterbank Energies

MFCC Features
~~~~~~~~~~~~~~

.. image:: https://github.com/astorfi/speech_feature_extraction/blob/master/images/Speech_GIF.gif 

The supported attributes for generating MFCC features can be seen by investigating the related function:

.. code-block:: python
      
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


Filterbank Energy Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

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
	   
log - Filterbank Energy Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The attributes for ``log_filterbank energies`` are the same for ``filterbank energies`` too.

.. code-block:: python

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
	    
Stack Frames
~~~~~~~~~~~~

In ``Stack_Frames`` function, the stack of frames will be generated from the signal.

.. code-block:: python

	def stack_frames(sig, sampling_frequency, frame_length=0.020, frame_stride=0.020, Filter=lambda x: numpy.ones((x,)),
                 zero_padding=True):
	    """Frame a signal into overlapping frames.
	    :param sig: The audio signal to frame of size (N,).
	    :param sampling_frequency: The sampling frequency of the signal.
	    :param frame_length: The length of the frame in second.
	    :param frame_stride: The stride between frames.
	    :param Filter: The time-domain filter for applying to each frame. By default it is one so nothing will be changed.
	    :param zero_padding: If the samples is not a multiple of frame_length(number of frames sample), zero padding will 
				 be done for generating last frame.
	    :returns: Array of frames. size: number_of_frames x frame_len.
	    """



Test Example
~~~~~~~~~~~~

The test example can be seen in ``test/test.py`` as below:

.. code-block:: python

	import scipy.io.wavfile as wav
	import numpy as np
	import speechpy

	file_name = 'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav'
	fs, signal = wav.read(file_name)
	signal = signal[:,0]

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




	
For ectracting the feature at first, the signal samples will be stacked into frames. The features are computed for each frame in the stacked frames collection.

Dependencies
=============

Two packages of ``Scipy`` and ``NumPy`` are the required dependencies which will be installed automatically by running the ``setup.py`` file.
