import torch
import librosa
import numpy as np
import sys
import os
from model import SimpleRNN

# --- Configuration ---
# MUST match the training configuration
INPUT_SIZE = 64
HIDDEN_SIZE = 80
NUM_LAYERS = 2
MAX_PAD_LEN = 70
MODEL_PATH = 'simple_rnn_model.pth'

# Labels strictly in the order observed during training
LABELS = ['adelante', 'atrás', 'izquierda', 'derecha', 'alto', 'ir', 'sí', 'no', 'uno', 'dos', 'tres', 'cuatro', 'cinco']

def preprocess_audio(audio_path):
    """
    Load and preprocess audio exactly like the training data.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Extract Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=INPUT_SIZE, n_fft=2048, hop_length=512)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Padding/Truncating
        if mel_spectrogram_db.shape[1] > MAX_PAD_LEN:
            mfcc = mel_spectrogram_db[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mel_spectrogram_db.shape[1]
            mfcc = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')

        # Convert to tensor: (batch_size, seq_len, input_size)
        # We add batch dimension (unsqueeze)
        features = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0) 
        return features
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict(audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    num_classes = len(LABELS)
    model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, device).to(device)
    
    # Load trained weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Train the model first.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set to evaluation mode

    # Preprocess
    input_tensor = preprocess_audio(audio_path)
    if input_tensor is None:
        return

    input_tensor = input_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Get the index of the max log-probability
        _, predicted_idx = torch.max(output.data, 1)
        
        predicted_label = LABELS[predicted_idx.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx.item()].item()

    print(f"File: {os.path.basename(audio_path)}")
    print(f"Prediction: {predicted_label.upper()} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_wav_file>")
    else:
        audio_file = sys.argv[1]
        predict(audio_file)
