"""
model.py — XylemAutoencoder (fixed for dynamic input shapes)
"""

import torch
import torch.nn as nn

class XylemAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
        )

        # This layer will be initialized dynamically
        self.flatten = nn.Flatten()
        self._latent_dim = latent_dim
        self._initialized = False

    def _initialize_layers(self, input_shape):
        """Dynamically determine flattened size and initialize dense layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            enc_out = self.encoder_conv(dummy)
            flat_dim = enc_out.numel()
        self.fc_enc = nn.Linear(flat_dim, self._latent_dim)
        self.fc_dec = nn.Linear(self._latent_dim, flat_dim)

        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # 128 -> 256
            nn.Sigmoid(),
        )

        self._flat_dim = flat_dim
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._initialize_layers(tuple(x.shape[1:]))

        z = self.encoder_conv(x)
        z = self.flatten(z)
        z = self.fc_enc(z)
        decoded = self.fc_dec(z)
        decoded = decoded.view(-1, 128, 16, 16)
        recon = self.decoder_deconv(decoded)
        return recon, z
