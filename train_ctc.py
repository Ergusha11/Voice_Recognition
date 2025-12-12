import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_ctc import CTCDataset, ctc_collate_fn
from model_ctc import CTCSpeechModel
from vocabulary import Vocabulary
import os

# Hyperparameters
INPUT_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
AUDIO_DIR = 'synthetic_phrases'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 0. Ensure data exists
if not os.path.exists(AUDIO_DIR) or len(os.listdir(AUDIO_DIR)) == 0:
    print(f"Please run 'python3 generate_phrases.py' first!")
    exit()

# 1. Setup Data
vocab = Vocabulary()
dataset = CTCDataset(AUDIO_DIR, n_mels=INPUT_SIZE)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ctc_collate_fn)

print(f"Dataset Size: {len(dataset)}")
print(f"Vocabulary Size: {len(vocab)} chars")

# 2. Setup Model
model = CTCSpeechModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(vocab)).to(device)

# 3. Loss and Optimizer
# blank=0 is our definition in vocabulary.py
criterion = nn.CTCLoss(blank=vocab.BLANK_IDX, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting CTC Training...")

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    
    for i, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Inputs needs to be (Seq_Len, Batch, Input_Size) for PyTorch CTCLoss usually, 
        # BUT our model outputs (Batch, Seq_Len, Class).
        # CTCLoss expects input: (Input_Sequence_Length, Batch_Size, Number_of_Classes)
        # So we transpose output of model:
        
        # Forward
        outputs = model(inputs) # (Batch, Time, Classes)
        outputs = outputs.permute(1, 0, 2) # (Time, Batch, Classes)
        
        # Loss
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
        
        # Validation / Demo Decoding
        with torch.no_grad():
            # Decode the last batch's first item
            # Output: (Time, Batch, Classes) -> Take batch 0 -> (Time, Classes)
            probs = outputs[:, 0, :] 
            max_probs, predicted_indices = torch.max(probs, dim=1) # (Time)
            
            # Simple Greedy Decode
            pred_text = vocab.int_to_text(predicted_indices.tolist())
            
            # Reconstruct true text (slicing from the concatenated targets tensor)
            # This is a bit tricky with concatenated targets, so we'll skip exact logging
            # or just rely on visual inspection of prediction consistency
            print(f"  Sample Pred: '{pred_text}'")

torch.save(model.state_dict(), 'ctc_phrase_model.pth')
print("Model saved to 'ctc_phrase_model.pth'")