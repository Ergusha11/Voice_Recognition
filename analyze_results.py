import torch
from torch.utils.data import DataLoader
from dataset import SpeechDataset
from model import SimpleRNN
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Configuration (Must match training) ---
INPUT_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MAX_PAD_LEN = 70
BATCH_SIZE = 32
MODEL_PATH = 'simple_rnn_model.pth'
AUDIO_DIR = 'synthetic_audio_data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze():
    print("Loading data for analysis...")
    # augment=False used to evaluate on the standard dataset
    dataset = SpeechDataset(AUDIO_DIR, max_pad_len=MAX_PAD_LEN, augment=False)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(dataset.int_to_label)
    labels_map = dataset.int_to_label
    
    # Initialize model
    model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, device).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Running inference on full dataset...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc*100:.2f}%")

    # --- Confusion Matrix ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_map, yticklabels=labels_map)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved 'confusion_matrix.png'")

    # --- Classification Report ---
    print("Generating Classification Report...")
    report = classification_report(all_labels, all_preds, target_names=labels_map, zero_division=0)
    
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    print("Saved 'classification_report.txt'")
    print("-" * 30)
    print(report)

if __name__ == "__main__":
    analyze()
