from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import librosa
import numpy as np
import os
import shutil
from model import SimpleRNN
from io import BytesIO

# --- Configuration ---
INPUT_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MAX_PAD_LEN = 70
MODEL_PATH = 'simple_rnn_model.pth'
LABELS = ['adelante', 'atrás', 'izquierda', 'derecha', 'alto', 'ir', 'sí', 'no', 'uno', 'dos', 'tres', 'cuatro', 'cinco']

# Initialize App
app = FastAPI(title="Voice Command Recognition API")

# Load Model (Global variable)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
    
    num_classes = len(LABELS)
    # Initialize the model structure (Must match training!)
    model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, device).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

def preprocess_audio(audio_path):
    """
    Load and preprocess audio from a file path.
    """
    try:
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

        features = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0) 
        return features
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

@app.get("/")
def read_root():
    return {"message": "Voice Recognition API is running. Use POST /predict to recognize commands."}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Save the uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Preprocess
        input_tensor = preprocess_audio(temp_filename)
        
        if input_tensor is None:
             raise HTTPException(status_code=400, detail="Could not process audio file.")
        
        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            
            # Since we have Identity output, apply softmax for probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_label = LABELS[predicted_idx.item()]
            conf_score = confidence.item()

        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": conf_score,
            "all_probabilities": {label: prob.item() for label, prob in zip(LABELS, probabilities[0])}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
