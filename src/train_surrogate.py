"""
train_surrogate.py
----------------------------------------
Train a small CNN surrogate f_theta:

    image [1,256,256]  →  [Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy]

using the dataset built by build_surrogate_dataset.py:

    results/surrogate_dataset.pt

Outputs:
    results/physics_surrogate.pth   (model weights)
    results/physics_surrogate_meta.pt  (metadata: metric_names, etc.)
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

DATA_PATH = "results/surrogate_dataset.pt"
MODEL_PATH = "results/physics_surrogate.pth"
META_PATH = "results/physics_surrogate_meta.pt"

DEVICE = torch.device("cpu")  # you're on CPU in this environment


# -------------------------
#  Small CNN regressor
# -------------------------
class SurrogateCNN(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        # Simple but reasonably expressive conv stack
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 32 -> 16
            nn.ReLU(inplace=True),
        )
        # 128 channels @ 16x16 = 128 * 16 * 16
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run src/build_surrogate_dataset.py first."
        )

    bundle = torch.load(DATA_PATH, map_location="cpu")
    images = bundle["images"].float()   # [N,1,256,256]
    metrics = bundle["metrics"].float() # [N,5]
    metric_names = bundle.get(
        "metric_names",
        ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"],
    )

    print(f"📦 Loaded surrogate dataset: images {images.shape}, metrics {metrics.shape}")
    print(f"   metric_names: {metric_names}")

    dataset = TensorDataset(images, metrics)

    # 80/20 train/val split
    N = len(dataset)
    val_size = max(1, int(0.2 * N))
    train_size = N - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = SurrogateCNN(out_dim=metrics.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs = 60

    for epoch in range(1, epochs + 1):
        # --------- Train ---------
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= train_size

        # --------- Val ---------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= val_size

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        # Simple "best checkpoint" save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(
                {
                    "metric_names": metric_names,
                    "train_size": train_size,
                    "val_size": val_size,
                },
                META_PATH,
            )
            print(f"   ✅ New best val loss, model saved → {MODEL_PATH}")

    print("🏁 Surrogate training complete.")
    print(f"Best val MSE: {best_val_loss:.6f}")
    print(f"Final model weights: {MODEL_PATH}")
    print(f"Meta saved:          {META_PATH}")


if __name__ == "__main__":
    main()
