# models_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAE1D(nn.Module):
    def __init__(self, seq_len=1024, channels=1, base_filters=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, base_filters, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L/2
            nn.Conv1d(base_filters, base_filters*2, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L/4
            nn.Conv1d(base_filters*2, base_filters*4, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        # decoder uses upsample+conv
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(base_filters*4, base_filters*2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(base_filters*2, base_filters, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(base_filters, channels, kernel_size=9, padding=4),
            # final linear activation (regression output)
        )

    def forward(self, x):
        # x shape (B, L) -> convert to (B, C, L)
        x = x.unsqueeze(1)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.squeeze(1)
        return out