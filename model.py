import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = True # Enable bidirectional

        # LSTM layer
        # bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)

        # Fully connected layer
        # If bidirectional, output size is hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden state and cell state
        # Shape: (num_layers * num_directions, batch, hidden_size)
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # out: (batch, seq_len, num_directions * hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        # For bidirectional, we usually concat the last hidden state of forward 
        # and the last hidden state of backward. 
        # But 'out' already contains concatenated features at each step.
        # We take the last time step.
        out = self.fc(out[:, -1, :])
        return out
