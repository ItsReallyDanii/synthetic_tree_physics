import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import XylemAutoencoder
from src.flow_simulation_utils import compute_flow_metrics  # physics proxy
from torchvision import transforms
from PIL import Image
from glob import glob
import csv

# ===========================
# 🔧 Config
# ===========================
DEVICE = "cpu"
EPOCHS = 100
LR = 1e-4
IMG_DIR = "data/generated_microtubes"  # all generated structures
MODEL_PATH = "results/model_hybrid.pth"
SAVE_PATH = "results/model_physics_tuned.pth"
LOG_PATH = "results/physics_training_log.csv"

# ===========================
# 🧱 Load Dataset
# ===========================
def load_dataset(path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),  # standardize all images
        transforms.ToTensor(),
    ])

    files = sorted(glob(os.path.join(path, "*.png")))
    imgs = []
    for f in files:
        img = Image.open(f)
        img = transform(img)
        # normalize to 0–1 explicitly
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        imgs.append(img)
    if not imgs:
        raise RuntimeError(f"No images found in {path}")
    return torch.stack(imgs)

# ===========================
# ⚙️ Compute Physics Loss
# ===========================
def compute_physics_loss(batch):
    K_vals, P_vals = [], []
    for img in batch:
        img_np = img.squeeze().detach().cpu().numpy()
        metrics = compute_flow_metrics(img_np, grad_scale=50.0)  # amplify flow sensitivity
        K_vals.append(metrics["K"])
        P_vals.append(metrics["Porosity"])
    K_mean = np.mean(K_vals)
    P_mean = np.mean(P_vals)
    # physics loss penalizes deviation from stable regime
    phys_loss = (1.0 - K_mean) ** 2 + (0.3 - P_mean) ** 2
    return torch.tensor(phys_loss, dtype=torch.float32, requires_grad=True), K_mean, P_mean

# ===========================
# 🚀 Training Loop
# ===========================
def main():
    print(f"🌱 Physics-informed fine-tuning started on {DEVICE}")

    # model load
    model = XylemAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Loaded pretrained model from {MODEL_PATH}")

    # dataset load
    imgs = load_dataset(IMG_DIR)
    print(f"🧩 Loaded {len(imgs)} generated structures.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    recon_loss_fn = nn.MSELoss()

    # log file setup
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "recon_loss", "phys_loss", "K_mean", "P_mean", "grad_norm"])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        imgs_batch = imgs.to(DEVICE)
        recon, _ = model(imgs_batch)

        recon_loss = recon_loss_fn(recon, imgs_batch)

        phys_loss, K_mean, P_mean = compute_physics_loss(recon)
        total_loss = recon_loss + 0.8 * phys_loss  # weighted combo

        total_loss.backward()

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | "
                  f"Total: {total_loss.item():.5f} | "
                  f"Recon: {recon_loss.item():.5f} | "
                  f"Phys: {phys_loss.item():.5f} | "
                  f"K: {K_mean:.5f} | "
                  f"Porosity: {P_mean:.5f} | "
                  f"GradNorm: {grad_norm:.2e}")

        # log every epoch
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch,
                             float(total_loss.item()),
                             float(recon_loss.item()),
                             float(phys_loss.item()),
                             float(K_mean),
                             float(P_mean),
                             float(grad_norm)])

    print("✅ Physics-informed fine-tuning complete.")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 Model saved → {SAVE_PATH}")
    print(f"🧾 Training log saved → {LOG_PATH}")


if __name__ == "__main__":
    main()
