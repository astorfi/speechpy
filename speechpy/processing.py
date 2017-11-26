import decimal
import numpy as np
import math


# 1.4 becomes 1 and 1.6 becomes 2. special case: 1.5 becomes 2.
def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def preemphasize(signal, cof=0.98):
    """preemphasising on the signal.
    :param signal: The input signal.
    :param coeff: The preemphasising coefficient. 0 equals to no filtering.
    :returns: the pre-emphasized signal.
    """

    rolled_signal = np.roll(signal, shift=1)
    return signal - cof * rolled_signal

def stack_frames(sig, sampling_frequency, frame_length=0.020, frame_stride=0.020, Filter=lambda x: np.ones((x,)),
                 zero_padding=True):
    """Frame a signal into overlapping frames.

    Args:
        sig (array): The audio signal to frame of size (N,).
        sampling_frequency (int): The sampling frequency of the signal.
        frame_length (float): The length of the frame in second.
        frame_stride (float): The stride between frames.
        Filter (array): The time-domain filter for applying to each frame. By default it is one so nothing will be changed.
        zero_padding (boolean): If the samples is not a multiple of frame_length(number of frames sample), zero padding will
                         be done for generating last frame.

    Returns:

        array: Array of frames of size (number_of_frames x frame_len).

    """

    # :param sig: The audio signal to frame of size (N,).
    # :param sampling_frequency: The sampling frequency of the signal.
    # :param frame_length: The length of the frame in second.
    # :param frame_stride: The stride between frames.
    # :param Filter: The time-domain filter for applying to each frame. By default it is one so nothing will be changed.
    # :param zero_padding: If the samples is not a multiple of frame_length(number of frames sample), zero padding will
    #                      be done for generating last frame.
    # :returns: Extracted_Frames: Array of frames. size: (number_of_frames x frame_len).

    ## Check dimension
    assert sig.ndim == 1, "Signal dimention should be of the format of (N,) but it is %s instead" % str(sig.shape)

    # Initial necessary values
    length_signal = sig.shape[0]
    frame_sample_length = int(np.round(sampling_frequency * frame_length))  # Defined by the number of samples
    frame_stride = float(np.round(sampling_frequency * frame_stride))

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


def fft_spectrum(frames, fft_points=512):
    """This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued
    array by means of an efficient algorithm called the Fast Fourier Transform (FFT). Please refer to
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html for further details.

    :param frames: The frame array in which each row is a frame.
    :param fft_points: The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    :param num_keep_coefficients: The number of coefficients that is kept.
    :returns: If frames is an num_frames x sample_per_frame matrix, output will be num_frames x FFT_LENGTH.
    """
    SPECTRUM_VECTOR = np.fft.rfft(frames, n=fft_points, axis=-1, norm=None)
    return np.absolute(SPECTRUM_VECTOR)


def power_spectrum(frames, fft_points=512):
    """Power spectrum of each frame.

    :param frames: The frame array in which each row is a frame.
    :param fft_points: The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    :returns: If frames is an num_frames x sample_per_frame matrix, output will be num_frames x fft_length.
    """
    return 1.0 / fft_points * np.square(fft_spectrum(frames, fft_points))


def log_power_spectrum(frames, fft_points=512, normalize=True):
    """Log power spectrum of each frame in frames.

    :param frames: The frame array in which each row is a frame.
    :param fft_points: The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    :param norm: If norm=1, the log power spectrum will be normalized.
    :returns: If frames is an num_frames x sample_per_frame matrix, output will be num_frames x fft_length.
    """
    power_spec = power_spectrum(frames, fft_points)
    power_spec[power_spec <= 1e-20] = 1e-20
    log_power_spec = 10 * np.log10(power_spec)
    if normalize:
        return log_power_spec - np.max(log_power_spec)
    else:
        return log_power_spec


def Derivative_Feature_Fn(feat, DeltaWindows):
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

        dif = Range * FEAT[:, offset + Range:offset + Range + cols] - FEAT[:, offset - Range:offset - Range + cols]
        Scale += 2 * np.power(Range, 2)
        DIF += dif

    return DIF / Scale


def cmvn(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and variance normalization
    (CMVN) on input feature vector "vec". The code assumes that there is one observation per row.

    :param vec: input feature matrix (size:(num_observation,num_features))
    :param variance_normalization: If the variance normilization should be performed or not.
    :return: The mean(or mean+variance) normalized feature vector.
    """
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / stdev_vec
    else:
        output = mean_subtracted

    return output


def cmvnw(vec, win_size=301, variance_normalization=False):
    """ This function is aimed to perform local cepstral mean and variance normalization on a sliding window.
    (CMVN) on input feature vector "vec". The code assumes that there is one observation per row.

    :param vec: input feature matrix (size:(num_observation,num_features))
    :param win_size: The size of sliding window for local normalization. Default=301 which is around 3s if 100 Hz rate is considered(== 10ms frame stide)
    :param variance_normalization: If the variance normilization should be performed or not.
    :return: The mean(or mean+variance) normalized feature vector.
    """
    # Get the shapes
    rows, cols = vec.shape

    # Windows size must be odd.
    assert type(win_size) == int, "Size must be of type 'int'!"
    assert win_size % 2 == 1, "Windows size must be odd!"

    # Padding and initial definitions
    pad_size = int((win_size - 1) / 2)
    vec_pad = np.lib.pad(vec, ((pad_size, pad_size), (0, 0)), 'symmetric')
    mean_subtracted = np.zeros(np.shape(vec), dtype=np.float32)

    for i in range(rows):
        window = vec_pad[i:i + win_size, :]
        window_mean = np.mean(window, axis=0)
        mean_subtracted[i, :] = vec[i, :] - window_mean

    # Variance normalization
    if variance_normalization:

        # Initial definitions.
        variance_normalized = np.zeros(np.shape(vec), dtype=np.float32)
        vec_pad_variance = np.lib.pad(mean_subtracted, ((pad_size, pad_size), (0, 0)), 'symmetric')

        # Looping over all observations.
        for i in range(rows):
            window = vec_pad_variance[i:i + win_size, :]
            window_variance = np.std(window, axis=0)
            variance_normalized[i, :] = mean_subtracted[i, :] / window_variance
        output = variance_normalized
    else:
        output = mean_subtracted

    return output



# def resample_Fn(wave, fs, f_new=16000):
#     """This function resample the data to arbitrary frequency
#     :param fs: Frequency of the sound file.
#     :param wave: The sound file itself.
#     :returns:
#            f_new: The new frequency.
#            signal_new: The new signal samples at new frequency.
#
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
