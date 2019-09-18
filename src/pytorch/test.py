import numpy as np
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# alpha = 0.1
# for i in range(100):
#     a = np.random.beta(alpha, alpha)
#     print(a)

audio = 'F:/task1/TAU-urban-acoustic-scenes-2019-development/audio/airport-barcelona-0-6-a.wav'
y, sr = librosa.load(audio)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_S = librosa.amplitude_to_db(S)
# plt.figure()
# librosa.display.specshow(log_S, sr=sr)
# plt.show()
print (y.shape)
y_h, y_p = librosa.effects.hpss(y)
print (y_h.shape)
print (y_p.shape)
S_h = librosa.feature.melspectrogram(y_h, sr=sr, n_mels=128)
log_S_h = librosa.amplitude_to_db(S_h)
# plt.figure()
# librosa.display.specshow(log_S_h, sr=sr)
# plt.show()

S_p = librosa.feature.melspectrogram(y_p, sr=sr, n_mels=128)
log_S_p = librosa.amplitude_to_db(S_p)
# plt.figure()
# librosa.display.specshow(log_S_p, sr=sr)
# plt.show()



