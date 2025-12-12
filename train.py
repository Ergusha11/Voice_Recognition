import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpeechDataset
from model import SimpleRNN
import os

# Hyperparameters
INPUT_SIZE = 64     # Must match n_mels in dataset.py
HIDDEN_SIZE = 80
NUM_LAYERS = 2
NUM_EPOCHS = 200    
BATCH_SIZE = 16     
LEARNING_RATE = 0.0005 # Increased slightly, relying on gradient clipping
AUDIO_DIR = 'synthetic_audio_data'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Data
print("Loading data with augmentation enabled...")
dataset = SpeechDataset(AUDIO_DIR, augment=True) # Enable augmentation
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.int_to_label}")

# 2. Initialize Model
num_classes = len(dataset.int_to_label)
model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, device).to(device)

# 3. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Training Loop
print("Starting training...")
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (Crucial for Vanilla RNN)
        # Clips gradients to have a max norm of 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

print("Training finished.")

# 5. Save the model
torch.save(model.state_dict(), 'simple_rnn_model.pth')
print("Model saved to 'simple_rnn_model.pth'")
