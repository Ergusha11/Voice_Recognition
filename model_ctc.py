import torch
import torch.nn as nn

class CTCSpeechModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_chars):
        """
        num_chars: Size of vocabulary (including blank)
        """
        super(CTCSpeechModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=self.bidirectional)
        
        # Output layer
        # Maps hidden state to character probabilities
        self.fc = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Size)
        
        # LSTM output: (Batch, Seq_Len, Hidden*2)
        out, _ = self.lstm(x)
        
        # Linear projection
        out = self.fc(out)
        
        # Log Softmax is required for CTCLoss (usually applied here or in Loss)
        # We apply it here for clarity. Dim=2 is the character dimension.
        out = torch.nn.functional.log_softmax(out, dim=2)
        
        return out