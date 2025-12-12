import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class SpeechDataset(Dataset):
    def __init__(self, audio_dir, max_pad_len=70, augment=False):
        self.audio_dir = audio_dir
        self.max_pad_len = max_pad_len
        self.augment = augment
        self.filepaths = []
        self.labels = []
        self.label_to_int = {}
        self.int_to_label = []

        # Collect all audio files and their labels
        # Assuming filenames are like "command_index_variation.wav" e.g., "adelante_0_1.wav"
        for filename in os.listdir(audio_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(audio_dir, filename)
                
                # Robust parsing strategy:
                # Split by '_'
                parts = filename.split('_')
                
                # If commands don't have '_', the label is parts[0]
                # If they do, we need a smarter way. But our commands list is simple.
                # Let's rely on the known structure: [Label]_[ClassIdx]_[VarIdx].wav
                # Actually, simpler: just take everything before the LAST TWO underscores.
                # BUT, wait. generate_data uses: f"{command}_{i}_{j}.wav"
                # If command is "adelante", filename is "adelante_0_0.wav". parts=["adelante", "0", "0.wav"]
                # Label is parts[0].
                
                label = parts[0]
                
                self.filepaths.append(filepath)
                self.labels.append(label)

                if label not in self.label_to_int:
                    self.label_to_int[label] = len(self.int_to_label)
                    self.int_to_label.append(label)

    def __len__(self):
        return len(self.filepaths)

    def _augment_audio(self, y):
        # 1. Add White Noise
        if random.random() < 0.5:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape[0])

        # 2. Time Shift
        if random.random() < 0.5:
            shift_amt = int(random.uniform(-0.2, 0.2) * y.shape[0])
            y = np.roll(y, shift_amt)
        
        # 3. Change Amplitude (Gain)
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            y = y * gain

        return y

    def __getitem__(self, idx):
        audio_path = self.filepaths[idx]
        label = self.labels[idx]
        numeric_label = self.label_to_int[label]

        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Apply augmentation if enabled
        if self.augment:
            y = self._augment_audio(y)

        # Extract Mel-spectrogram features
        # n_mels: number of Mel bands
        # n_fft: length of the FFT window
        # hop_length: number of samples between successive frames
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max) # Convert to dB scale

        # Padding/Truncating to a fixed length
        if mel_spectrogram_db.shape[1] > self.max_pad_len:
            mfcc = mel_spectrogram_db[:, :self.max_pad_len]
        else:
            pad_width = self.max_pad_len - mel_spectrogram_db.shape[1]
            mfcc = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')

        # Convert to PyTorch tensor
        # RNN expects (batch, seq_len, input_size)
        # Our mfcc is (n_mels, time_steps) -> Transpose to (time_steps, n_mels)
        features = torch.tensor(mfcc.T, dtype=torch.float32) 

        return features, torch.tensor(numeric_label, dtype=torch.long)