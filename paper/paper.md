---
title: 'SpeechPy - A Library for Speech Processing and Recognition'
tags:
  - Python
authors:
  - name: Amirsina Torfi
    orcid: 0000-0003-2282-4361
    affiliation: "1"
affiliations:
 - name: Virginia Tech, Department of Computer Science
   index: 1
date: 15 May 2018
bibliography: paper.bib
---

# Abstract
SpeechPy is an open source Python package that contains speech preprocessing techniques, speech features, and important post-processing operations. It provides most frequent used speech features including MFCCs and filterbank energies alongside with the log-energy of filter-banks. The aim of the package is to provide researchers with a simple tool for speech feature extraction and processing purposes in applications such as Automatic Speech Recognition and Speaker Verification.

# Introduction
Automatic Speech Recognition (ASR) requires three main components for
further analysis: Preprocessing, feature extraction, and
post-processing. Feature extraction, in an abstract meaning, is
extracting descriptive features from raw signal for speech
classification purposes. Due to the high
dimensionality, the raw signal can be less informative compared to
extracted higher level features. Feature extraction comes to our rescue
for turning the high dimensional signal to a lower dimensional and yet
a more informative version of that for sound recognition and
classification [@furui1986speaker; @guyon2008feature; @hirsch2000aurora].

![Scheme of speech recognition](_imgs/Scheme_of_speech_recognition_system.png)

Feature extraction, in essence, should be done considering the specific
application at hand. For example, in ASR applications, the linguistic
characteristics of the raw signal are of great importance and the other
characteristics must be
ignored [@yu2016automatic; @rabiner1993fundamentals]. On the other hand,
in Speaker Recognition (SR) task, solely voice-associated information
must be contained in the extracted feature [@campbell1997speaker]. So the
feature extraction goal is to extract the relevant feature from the raw
signal and map it to a lower dimensional feature space. The problem of
feature extraction has been investigated in pattern classification aimed
at preventing the curse of dimensionality. There are some feature
extraction approaches based on information theory
[@torfi2017construction; @shannon2001mathematical] applied to multimodal
signals and demonstrated promising results [@gurban2009information].

The speech features can be categorized into two general types of
acoustic and linguistic features. The former one is mainly related to
non-verbal sounds and the later one is associated with ASR and SR
systems for which verbal part has the major role. Perhaps one of the most
famous linguistic feature which is hard to beat is the Mel-Frequency
Cepstral Coefficients (MFCC). It uses speech raw frames in the range
from 20ms to 40ms for having stationary
characteristics [@rabiner1993fundamentals]. MFCC is widely used for both
ASR and SR tasks and more recently in the associated deep learning
applications as the input to the network rather than directly feeding
the signal [@deng2013recent; @lee2009unsupervised; @yu2011improved].
With the advent of deep learning [@lecun2015deep; @torfi2018attention],
major improvements have been achieved by using deep neural networks
rather than traditional methods for speech recognition
applications [@variani2014deep; @hinton2012deep; @liu2015deep].

With the availability of free software for speech recognition such as
VOICEBOX, most of these softwares are Matlab-based which limits
their reproducibility due to commercial issues. Another great package is
PyAudioAnalysis [@giannakopoulos2015pyaudioanalysis], which is a
the comprehensive package developed in Python. However, the issue with
PyAudioAnalysis is that its complexity and being too verbose for
extracting simple features and it also lacks some important
preprocessing and post-processing operations for its current version.

Considering the recent advent of deep learning in ASR and SR and the
importance of the accurate speech feature extraction, here are the
motivations behind SpeechPy package:

  * Developing a free open source package which covers important
    preprocessing techniques, speech features, and post-processing
    operations required for ASR and SR applications.

  * A simple package with a minimum degree of complexity should be
    available for beginners.

  * A well-tested and continuously integrated package for future
    developments should be developed.

SpeechPy has been developed to satisfy the aforementioned needs. It
contains the most important preprocessing and post-processing operations
and a selection of frequently used speech features. The package is free
and released as an open source software. Continuous integration
using for instant error check and validity of changes has been deployed
for SpeechPy. Moreover, prior to the latest official release of
SpeechPy, the package has successfully been utilized for research
purposes [@torfi20173d; @torfi2017text].

# Package Eco-system


SpeechPy has been developed using Python language for its interface and
backed as well. An empirical study demonstrated that Python as a
scripting language, is more effective and productive than conventional
languages for some programming problems and memory consumption is
often “better than Java and not much worse than C or
C++” [@prechelt2000empirical]. We chose Python due to its simplicity and
popularity. Third-party libraries are avoided except *Numpy* and *Scipy*
for handling data and numeric computations.

## Complexity

As the user should not and does not even need to manipulate the internal
package structure, object-oriented programming is mostly used for
package development which provides an easier interface for the user with a
sacrifice to the simplicity of the code. However, the internal code
complexity of the package does not affect the user experience since the
modules can easily be called with the associated arguments. SpeechPy is
a library with a collection of sub-modules.

## Code Style and Documentation

SpeechPy is constructed based on PEP 8 style guide for Python codes.
Moreover, it is extensively documented using the formatted docstrings
and Sphinx for further automatic modifications to the document in
case of changing internal modules. The full documentation of the project
will be generated in HTML and PDF format using Sphinx and is hosted
online. The official releases of the project are hosted on the Zenodo as
well [@torfispeechpy].

![A general view of the package](_imgs/packageview.png)

## Continuous Testing and Extensibility

The output of each function has been evaluated as well as using different
tests as opposed to the other existing standard packages. For continuous
testing, the code is hosted on GitHub and integrated with Travis CI.
Each modification to the code must pass the unit tests defined for the
continuous integration. This will ensure the package does not break with
unadapted code scripts. However, the validity of the modifications
should always be investigated with the owner or authorized collaborators
of the project. The code will be tested at each time of modification for
Python versions *“2.7”*, *“3.4”* and *“3.5”*. In the future, these
versions are subject to change.

![Travic CI web interface after testing SpeechPy against a new change](_imgs/travicCI.png)

# Availability

## Operating system {#operating-system .unnumbered}

Tested on Ubuntu 14.04 and 16.04 LTS Linux, Apple Mac OS X 10.9.5 , and
Microsoft Windows 7 & 10. We expect that SpeechPy works on any
distribution as long as Python and the package dependencies are
installed.

## Programming language {#programming-language .unnumbered}

The package has been tested with Python 2.7, 3.4 and 3.5. However, using
Python 3.5 is suggested.

## Additional system requirements & dependencies {#additional-system-requirements-dependencies .unnumbered}

SpeechPy is a light package and small computational power would be
enough for running it. Although the speed of the execution is totally
dependent on the system architecture. The dependencies are as follows:

  * Numpy

  * SciPy

# Acknowledgement

This work has been completed with computational resources provided by the West Virginia University and Virginia Tech and is based upon a work
supported by the Center for Identification Technology Research (CITeR) and the National Science Foundation (NSF) under Grant \#1650474.
I would like to thank professor Nasser Nasrabadi for supporting me through this project and for his valuable supervision regarding my research in speech technology.

# References
