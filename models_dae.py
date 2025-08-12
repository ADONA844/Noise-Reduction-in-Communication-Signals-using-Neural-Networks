# models_dae.py
import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, seq_len=1024, latent_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, seq_len//2),
            nn.ReLU(),
            nn.Linear(seq_len//2, seq_len//4),
            nn.ReLU(),
            nn.Linear(seq_len//4, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, seq_len//4),
            nn.ReLU(),
            nn.Linear(seq_len//4, seq_len//2),
            nn.ReLU(),
            nn.Linear(seq_len//2, seq_len),
        )

    def forward(self, x):
        # x shape (B, L)
        z = self.encoder(x)
        out = self.decoder(z)
        return out