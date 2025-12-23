import torch
import librosa
import numpy as np
import sys
import os
from model_ctc import CTCSpeechModel
from vocabulary import Vocabulary
import preprocessing_ctc

# Configuration
INPUT_SIZE = 13
HIDDEN_SIZE = 30
NUM_LAYERS = 1
MODEL_PATH = 'ctc_phrase_model.pth'

def predict_phrase(audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Vocabulary & Model
    vocab = Vocabulary()
    num_chars = len(vocab)
    
    model = CTCSpeechModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_chars).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' not found.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Process Audio
    try:
        # Use centralized methodology pipeline
        features = preprocessing_ctc.preprocess_pipeline(audio_path)
        
        # Add batch dimension: (1, Time, n_mfcc)
        input_tensor = features.unsqueeze(0).to(device)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return

    # 3. Inference
    with torch.no_grad():
        # Output: (Batch, Time, Classes)
        outputs = model(input_tensor)
        
        # Greedy Decode: Take max probability at each time step
        # Shape: (1, Time)
        _, predicted_indices = torch.max(outputs, dim=2)
        
        # Convert indices to text
        # .squeeze() removes the batch dimension -> (Time)
        transcription = vocab.int_to_text(predicted_indices.squeeze().tolist())
        
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Prediction: '{transcription}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_ctc.py <path_to_wav_file>")
    else:
        predict_phrase(sys.argv[1])
