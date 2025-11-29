"""
train.py — trains the convolutional autoencoder on synthetic xylem-like images.
Robust path handling + automatic dependency install + uniform image resizing.
"""

import os, sys, subprocess

# --- Ensure dependencies are present ---
REQUIRED = ["torch", "torchvision", "numpy", "matplotlib", "Pillow"]
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# --- Make sure src/ is importable no matter where we run from ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- Standard imports ---
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# --- Import model safely ---
try:
    from src.model import XylemAutoencoder
except ModuleNotFoundError:
    try:
        import model
        XylemAutoencoder = model.XylemAutoencoder
    except Exception as e:
        raise ImportError(
            f"Could not import XylemAutoencoder. "
            f"Check that {ROOT_DIR}/src/model.py exists.\nOriginal error: {e}"
        )

# --- Config ---
DATA_DIR = os.path.join(ROOT_DIR, "data", "generated_microtubes")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
LATENT_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset ---
class XylemDataset(Dataset):
    def __init__(self, path):
        self.files = glob.glob(os.path.join(path, "*.png"))
        if not self.files:
            raise FileNotFoundError(f"No PNG images found in {path}")
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),   # ensure uniform input size
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        return self.transform(img)

# --- Train loop ---
def train():
    dataset = XylemDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = XylemAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {len(dataset)} images for {EPOCHS} epochs using {DEVICE}")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

        # Save reconstruction preview
        with torch.no_grad():
            recon_img = recon[0].cpu().numpy().squeeze()
            orig_img = batch[0].cpu().numpy().squeeze()
            fig, axs = plt.subplots(1, 2, figsize=(4, 2))
            axs[0].imshow(orig_img, cmap="gray"); axs[0].set_title("Original")
            axs[1].imshow(recon_img, cmap="gray"); axs[1].set_title("Reconstructed")
            for ax in axs: ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"recon_epoch_{epoch+1}.png"))
            plt.close()

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "xylem_autoencoder.pt"))
    print(f"✅ Training complete. Results saved to {RESULTS_DIR}")

# --- Entry point ---
if __name__ == "__main__":
    train()
