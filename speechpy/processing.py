import decimal
import numpy as np
import math

# 1.4 becomes 1 and 1.6 becomes 2. special case: 1.5 becomes 2.
def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def stack_frames(sig, sampling_frequency, frame_length=0.020, frame_stride=0.020, Filter=lambda x: numpy.ones((x,)),
                 zero_padding=False):
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

    ## Check dimension
    assert sig.ndim == 1, "Signal dimention should be of the format of (N,) but it is %s instead" % str(sig.shape)

    # Initial necessary values
    length_signal = sig.shape[0]
    frame_sample_length = int(np.round(sampling_frequency * frame_length))  # Defined by the number of samples
    frame_stride = float(np.round(sampling_frequency * frame_stride))

    # Check the feasibility of stacking
    if length_signal <= frame_sample_length:
        numframes = 1
    else:
        # Zero padding is done for allocating space for the last frame.
        if zero_padding:
            # Calculation of number of frames
            numframes = 1 + int(math.ceil((length_signal - frame_sample_length) / frame_stride))

            # Zero padding
            len_sig = int((numframes - 1) * frame_stride + frame_sample_length)
            additive_zeros = np.zeros((len_sig - length_signal,))
            signal = np.concatenate((sig, additive_zeros))

        else:
            # No zero padding! The last frame which does not have enough
            # samples(remaining samples <= frame_sample_length), will be dropped!
            numframes = 1 + int(math.floor((length_signal - frame_sample_length) / frame_stride))

            # new length
            len_sig = int((numframes - 1) * frame_stride + frame_sample_length)
            signal = sig[0:len_sig]


    # Getting the indices of all frames.
    indices = np.tile(np.arange(0, frame_sample_length), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_stride, frame_stride), (frame_sample_length, 1)).T
    indices = np.array(indices, dtype=np.int32)

    # Extracting the frames based on the allocated indices.
    frames = signal[indices]

    # Apply the windows function
    window = np.tile(Filter(frame_sample_length), (numframes, 1))
    Extracted_Frames = frames * window
    return Extracted_Frames


def fft_spectrum(frames, fft_length=512):
    """This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued
       array by means of an efficient algorithm called the Fast Fourier Transform (FFT).(ref: numpy documentation)
       please refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html for further details.

    :param frames: The frame array in which each row is a frame.
    :param fft_length: The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    :param num_keep_coefficients: The number of coefficients that is kept.
    :returns: If frames is an num_frames x sample_per_frame matrix, output will be num_frames x FFT_LENGTH.
    """
    SPECTRUM_VECTOR = np.fft.rfft(frames, n=fft_length, axis=-1, norm=None)
    return np.absolute(SPECTRUM_VECTOR)


def power_spectrum(frames, fft_length=512):
    """Power spectrum of each frame.

    :param frames: The frame array in which each row is a frame.
    :param fft_length: The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    :returns: If frames is an num_frames x sample_per_frame matrix, output will be num_frames x fft_length.
    """
    return 1.0 / fft_length * np.square(fft_spectrum(frames, fft_length))


def log_power_spectrum(frames, fft_length=512, normalize=True):
    """Log power spectrum of each frame in frames.

    :param frames: The frame array in which each row is a frame.
    :param fft_length: The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    :param norm: If norm=1, the log power spectrum will be normalized.
    :returns: If frames is an num_frames x sample_per_frame matrix, output will be num_frames x fft_length.
    """
    power_spec = power_spectrum(frames, fft_length)
    power_spec[power_spec <= 1e-20] = 1e-20
    log_power_spec = 10 * np.log10(power_spec)
    if normalize:
        return log_power_spec - np.max(log_power_spec)
    else:
        return log_power_spec

def Derivative_Feature_Fn(feat,DeltaWindows):
    """This function the derivative features.
    :param feat: The main feature vector(For returning the second order derivative it can be first-order derivative).
    :param DeltaWindows: The value of  DeltaWindows is set using the configuration parameter DELTAWINDOW.
    :returns:
           A NUMFRAMESxNUMFEATURES numpy array which is the derivative features along the features.
    """

    # Getting the shape of the vector.
    rows, cols = feat.shape

    # Difining the vector of differences.
    DIF = np.zeros(feat.shape, dtype=float)
    Scale = 0

    # Pad only along features in the vector.
    FEAT = np.lib.pad(feat, ((0, 0), (DeltaWindows, DeltaWindows)), 'edge')
    for i in range(DeltaWindows):

        # Start index
        offset = DeltaWindows

        # The dynamic range
        Range = i + 1

        dif = Range * FEAT[:,offset+Range:offset+Range+cols] - FEAT[:,offset-Range:offset-Range+cols]
        Scale += 2 * np.power(Range,2)
        DIF += dif

    return DIF/Scale


# def resample_Fn(wave, fs, f_new=16000):
#     """This function resample the data to arbitrary frequency
#     :param fs: Frequency of the sound file.
#     :param wave: The sound file itself.
#     :returns:
#            f_new: The new frequency.
#            signal_new: The new signal samples at new frequency.

#     dependency: from scikits.samplerate import resample
#     """
#
#     # Resampling using interpolation(There are other methods than 'sinc_best')
#     signal_new = resample(wave, float(f_new) / fs, 'sinc_best')
#
#     # Necessary data converting for saving .wav file using scipy.
#     signal_new = np.asarray(signal_new, dtype=np.int16)
#
#     # # Uncomment if you want to save the audio file
#     # # Save using new format
#     # wav.write(filename='resample_rainbow_16k.wav',rate=fr,data=signal_new)
#     return signal_new, f_new







