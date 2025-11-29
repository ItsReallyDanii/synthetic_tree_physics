import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

from src.model import XylemAutoencoder
from src.flow_simulation_utils import compute_flow_metrics, compute_batch_flow_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/model_hybrid.pth"
SAVE_PATH = "results/model_physics_tuned.pth"
LOG_PATH = "results/physics_training_log.csv"

DATA_DIR = "data/generated_microtubes"
os.makedirs("results", exist_ok=True)


# ===============================================================
# 1. Load and normalize dataset
# ===============================================================
def load_generated_structures(limit=None, size=(256, 256)):
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if limit is not None:
        files = files[:limit]

    imgs = []
    for f in files:
        img = Image.open(os.path.join(DATA_DIR, f)).convert("L").resize(size, Image.BICUBIC)
        arr = np.array(img, dtype=np.float32)  # ✅ force float32 here
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        # optional normalization cleanup
        if arr.mean() < 0.5:
            arr = 1 - arr
        arr = np.clip((arr - 0.5) * 2.0 + 0.5, 0, 1)
        arr = np.where(arr > 0.6, 1.0, 0.0)

        imgs.append(arr.astype(np.float32))  # ✅ make sure it's float32

    imgs = np.stack(imgs, axis=0).astype(np.float32)
    imgs = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)  # ✅ force float32 in torch
    return imgs


# ===============================================================
# 2. Physics loss (permeability + porosity)
# ===============================================================
def compute_physics_loss(batch):
    K_mean, P_mean = compute_batch_flow_metrics(batch)
    K_loss = 1.0 - K_mean
    P_loss = (0.4 - P_mean) ** 2
    return torch.tensor(K_loss, dtype=torch.float32), torch.tensor(P_loss, dtype=torch.float32)


# ===============================================================
# 3. Training loop
# ===============================================================
def main():
    print(f"🌱 Physics-informed fine-tuning started on {DEVICE}")

    model = XylemAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    print(f"✅ Loaded pretrained model from {MODEL_PATH}")

    imgs = load_generated_structures(limit=None, size=(256, 256)).to(DEVICE)
    print(f"🧩 Loaded {len(imgs)} generated structures.")

    recon_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    λ_phys = 1e-4
    EPOCHS = 100
    log = []

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        recon, latent = model(imgs)

        recon_loss = recon_loss_fn(recon, imgs)
        K_loss, P_loss = compute_physics_loss(recon)
        phys_loss = K_loss + P_loss
        total_loss = recon_loss + λ_phys * phys_loss

        total_loss.backward()
        optimizer.step()

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        grad_norm = grad_norm / (len(list(model.parameters())) + 1e-8)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{EPOCHS} | Total: {total_loss.item():.5f} | "
                f"Recon: {recon_loss.item():.5f} | Phys: {phys_loss.item():.5f} | "
                f"K: {K_loss.item():.5f} | Porosity: {P_loss.item():.5f} | GradNorm: {grad_norm:.5e}"
            )

        log.append({
            "epoch": epoch,
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "phys": phys_loss.item(),
            "K": K_loss.item(),
            "Porosity": P_loss.item(),
            "GradNorm": grad_norm
        })

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 Model saved → {SAVE_PATH}")

    import csv
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    print(f"🧾 Training log saved → {LOG_PATH}")


if __name__ == "__main__":
    main()
