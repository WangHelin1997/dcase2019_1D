import librosa
from SpecAugment import spec_augment_pytorch

audio_path = 'F:/task1/TAU-urban-acoustic-scenes-2019-development/audio/airport-barcelona-0-6-a.wav'
audio, sampling_rate = librosa.load(audio_path)
mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sampling_rate,n_mels=128,hop_length=1024,fmax=14000)
warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram, alpha=0.01)
spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram,
                                               title="pytorch Warped & Masked Mel Spectrogram")
