from speech_feature_extraction import stack_signal
from speech_feature_extraction import feature_extraction
import scipy.io.wavfile as wav

file_name = 'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav'
fs, signal = wav.read(file_name)
signal = signal[:,0]
stacked_signal = stack_signal(signal, fs, frame_length = 0.020, overlap_factor = 0.0)
feature = feature_extraction(stacked_signal, fs, feature_type='logfbank_energy')
print(feature.shape)
