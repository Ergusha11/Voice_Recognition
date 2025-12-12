import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from vocabulary import Vocabulary

class CTCDataset(Dataset):
    def __init__(self, audio_dir, n_mels=64):
        self.audio_dir = audio_dir
        self.files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.vocab = Vocabulary()
        self.n_mels = n_mels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.audio_dir, file_name)
        
        # Parse label from filename: "ir_adelante@12_0.wav" -> "ir adelante"
        # We split by '@' as defined in generator
        text_part = file_name.split('@')[0] 
        label_text = text_part.replace("_", " ")

        # 1. Process Audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Mel Spectrogram
        # Transpose so shape is (Seq_Len, Input_Size)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # (N_Mels, Time) -> (Time, N_Mels)
        mel_spec_db = mel_spec_db.T 
        
        # Convert to Tensor
        audio_tensor = torch.tensor(mel_spec_db, dtype=torch.float32)

        # 2. Process Text
        target_indices = self.vocab.text_to_int(label_text)
        target_tensor = torch.tensor(target_indices, dtype=torch.long)

        return audio_tensor, target_tensor

def ctc_collate_fn(batch):
    """
    Custom collate function to handle variable length audio and text.
    Returns:
        inputs: Padded audio (Batch, Max_Time, n_mels)
        targets: Concatenated targets (Total_Len)
        input_lengths: Tuple of actual audio lengths
        target_lengths: Tuple of actual text lengths
    """
    audio_tensors, target_tensors = zip(*batch)

    # 1. Pad Audio
    # Find max length in this batch
    max_audio_len = max([x.size(0) for x in audio_tensors])
    n_mels = audio_tensors[0].size(1)
    
    # Create padded batch tensor
    batch_size = len(audio_tensors)
    inputs = torch.zeros(batch_size, max_audio_len, n_mels)
    input_lengths = []

    for i, tensor in enumerate(audio_tensors):
        seq_len = tensor.size(0)
        inputs[i, :seq_len, :] = tensor
        input_lengths.append(seq_len)

    # 2. Process Targets for CTCLoss
    # CTCLoss expects targets to be a 1D concatenated tensor of all labels in batch
    targets = torch.cat(target_tensors)
    target_lengths = [len(t) for t in target_tensors]

    return inputs, targets, tuple(input_lengths), tuple(target_lengths)