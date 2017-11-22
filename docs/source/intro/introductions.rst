Introduction
============

The purpose of this project is to provide a package for speech processing and
feature extraction. This library provides most frequent used speech features including MFCCs and filterbank energies alongside with the log-energy of filterbanks.


.. image:: ../_static/img/speech.gif
   :height: 200px
   :width: 400 px
   :scale: 100 %
   :alt: alternate text
   :align: center

-----------
Motivation
-----------

There are different motivations for this open source project.

~~~~~~~~~~~~~~~~~~~~~~~~~
Deep Learning application
~~~~~~~~~~~~~~~~~~~~~~~~~

One of the main reasons for creating this package was to provide necessary features for deep learning applications such as ASR(Automatic Speech Recognition) or SR(Speaker Recognition).
As a results, most of the features that are necessary are provided hear.

~~~~~~~~~~~~~~~~~~~
Pythonic Packaging
~~~~~~~~~~~~~~~~~~~

Another reason for creating this package was to have a Pythonic environment for
speech recognition and feature extraction due to the fact that the Python language
is becoming ubiquotous!


----------------------
How to Install?
----------------------

There are two possible ways for installation of this package: local installation and PyPi.

~~~~~~~~~~~~~~~~~~~
Local Installation
~~~~~~~~~~~~~~~~~~~

For local installation at first the repository must be cloned::

	  git clone https://github.com/astorfi/speech_feature_extraction.git


After cloning the reposity, root to the repository directory then execute::

    python setup.py develop

~~~~~
Pypi
~~~~~

The package is available on PyPi. For direct installation simply execute the following:


.. code-block:: shell

     pip install speechpy

--------
Citation
--------

If you used this package, please cite it as follows:

.. code:: bash

	    @misc{amirsina_torfi_2017_840395,
         		author = {Amirsina Torfi},
        		title = {{SpeechPy: Speech recognition and feature extraction}},
         		month = aug,
         		year = 2017,
        		doi = {10.5281/zenodo.840395},
                url = {https://doi.org/10.5281/zenodo.840395}
            }
