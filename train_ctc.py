import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_ctc import CTCDataset, ctc_collate_fn
from model_ctc import CTCSpeechModel
from vocabulary import Vocabulary
import os
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
INPUT_SIZE = 13     # MFCC features
HIDDEN_SIZE = 30
NUM_LAYERS = 2
NUM_EPOCHS = 120
BATCH_SIZE = 12
LEARNING_RATE = 0.0016
AUDIO_DIR = 'synthetic_phrases'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 0. Ensure data exists
if not os.path.exists(AUDIO_DIR) or len(os.listdir(AUDIO_DIR)) == 0:
    print(f"Please run 'python3 generate_phrases.py' first!")
    exit()

# 1. Setup Data with Train/Validation Split
vocab = Vocabulary()
full_dataset = CTCDataset(AUDIO_DIR, augment=True)

# Split 80% Train, 20% Validation
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Total Dataset: {total_size}")
print(f"Training Samples: {train_size}")
print(f"Validation Samples: {val_size}")
print(f"Vocabulary Size: {len(vocab)} chars")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ctc_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ctc_collate_fn)

# 2. Setup Model
model = CTCSpeechModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(vocab)).to(device)

# 3. Loss and Optimizer
criterion = nn.CTCLoss(blank=vocab.BLANK_IDX, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting CTC Training with Validation...")

# Lists to store metrics
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    
    for inputs, targets, input_lengths, target_lengths in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward
        outputs = model(inputs) # (Batch, Time, Classes)
        outputs = outputs.permute(1, 0, 2) # (Time, Batch, Classes)
        
        # Loss
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            outputs = outputs.permute(1, 0, 2)
            
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # --- Logging ---
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Simple Demo Decoding from Validation Set
        with torch.no_grad():
            if inputs.size(1) > 0: # Check if batch is valid
                probs = outputs[:, 0, :] 
                _, predicted_indices = torch.max(probs, dim=1)
                pred_text = vocab.int_to_text(predicted_indices.tolist())
                print(f"  Val Sample Pred: '{pred_text}'")

# 4. Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='--')
plt.title('Training vs Validation Loss (Overfitting Check)')
plt.xlabel('Epochs')
plt.ylabel('CTC Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_overfitting_check.png')
print("Graph saved to 'training_overfitting_check.png'")

# 5. Save Model
torch.save(model.state_dict(), 'ctc_phrase_model.pth')
print("Model saved to 'ctc_phrase_model.pth'")
