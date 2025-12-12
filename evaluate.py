import torch
from torch.utils.data import DataLoader
from dataset import SpeechDataset
from model import SimpleRNN
import os

# Configuration (Must match training)
INPUT_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MAX_PAD_LEN = 70  # Updated to 70
BATCH_SIZE = 16
MODEL_PATH = 'simple_rnn_model.pth'
AUDIO_DIR = 'synthetic_audio_data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model():
    print("Loading data for evaluation...")
    # Note: augment=False because we want to test on "clean" (or standard) data, 
    # though since all our data is synthetic/augmented, this just turns off the *extra* random noise for testing.
    dataset = SpeechDataset(AUDIO_DIR, max_pad_len=MAX_PAD_LEN, augment=False)
    test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(dataset.int_to_label)
    
    # Initialize model
    model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, device).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set to evaluation mode (turns off dropout, etc.)

    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # THE DECODER PART:
            # The model outputs raw logits (Identity function).
            # We use Argmax to find the index with the highest score.
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('----------------------------------------------------------------')
    print(f'Overall Accuracy: {100 * correct / total:.2f}%')
    print('----------------------------------------------------------------')
    
    print('Accuracy per Class:')
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'{dataset.int_to_label[i]:>12}: {100 * class_correct[i] / class_total[i]:.0f}%')
        else:
            print(f'{dataset.int_to_label[i]:>12}: N/A')

if __name__ == '__main__':
    evaluate_model()
