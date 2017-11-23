============
test
============

-------------
Test Package
-------------
Once the package has been installed, a test file can be directly run to show the results.
The test example can be seen in ``test/test_package.py`` as below:

.. code-block:: python

    import scipy.io.wavfile as wav
    import numpy as np
    import speechpy
    import os

    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav')
    fs, signal = wav.read(file_name)
    signal = signal[:,0]

    ############# Extract MFCC features #############
    mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
               num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    mfcc_cmvn = speechpy.cmvnw(mfcc,win_size=301,variance_normalization=True)
    print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

    mfcc_feature_cube = speechpy.extract_derivative_feature(mfcc)
    print('mfcc feature cube shape=', mfcc_feature_cube.shape)

    ############# Extract logenergy features #############
    logenergy = speechpy.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
               num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    logenergy_feature_cube = speechpy.extract_derivative_feature(logenergy)
    print('logenergy features=', logenergy.shape)

    # Example of staching frames
    signal = speechpy.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, Filter=lambda x: np.ones((x,)),
           zero_padding=True)

-----------
Test Local
-----------

There is an alternative local way of testing without the necessity to package installation.
The local test example can be found in ``test/test_package.py`` as follows:

.. code-block:: python

    import scipy.io.wavfile as wav
    import numpy as np
    import os
    import sys
    lib_path = os.path.abspath(os.path.join('..'))
    print(lib_path)
    sys.path.append(lib_path)
    import speechpy
    import os

    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav')
    fs, signal = wav.read(file_name)
    signal = signal[:,0]

    ############# Extract MFCC features #############
    mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                 num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    mfcc_cmvn = speechpy.cmvnw(mfcc,win_size=301,variance_normalization=True)
    print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

    mfcc_feature_cube = speechpy.extract_derivative_feature(mfcc)
    print('mfcc feature cube shape=', mfcc_feature_cube.shape)

    ############# Extract logenergy features #############
    logenergy = speechpy.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                 num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    logenergy_feature_cube = speechpy.extract_derivative_feature(logenergy)
    print('logenergy features=', logenergy.shape)

    # Example of staching frames
    signal = speechpy.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, Filter=lambda x: np.ones((x,)),
             zero_padding=True)



For ectracting the feature at first, the signal samples will be stacked into frames. The features are computed for each frame in the stacked frames collection.

-------------
Dependencies
-------------

Two packages of ``Scipy`` and ``NumPy`` are the required dependencies which will be installed automatically by running the ``setup.py`` file.
