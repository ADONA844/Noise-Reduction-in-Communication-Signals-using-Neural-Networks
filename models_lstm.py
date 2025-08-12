# models_lstm.py
import torch
import torch.nn as nn

class LstmDenoiser(nn.Module):
    def __init__(self, seq_len=1024, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)  # map hidden -> output scalar per time-step

    def forward(self, x):
        # x shape (B, L)
        x = x.unsqueeze(-1)  # (B, L, 1)
        h, _ = self.lstm(x)  # (B, L, hidden_size)
        out = self.fc(h)     # (B, L, 1)
        out = out.squeeze(-1)  # (B, L)
        return out