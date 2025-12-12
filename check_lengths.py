import librosa
import os
import numpy as np

audio_dir = "synthetic_audio_data"
lengths = []

print("Checking spectrogram lengths...")
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_dir, filename)
        y, sr = librosa.load(filepath, sr=None)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
        lengths.append(mel.shape[1])

print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Mean length: {np.mean(lengths)}")
