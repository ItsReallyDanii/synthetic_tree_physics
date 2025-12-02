import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SURROGATE_DATASET_PATH = "results/surrogate_dataset.pt"
OUT_PATH = "results/physics_surrogate.pth"


class PhysicsSurrogateCNN(nn.Module):
    """
    Simple CNN regressor:
      input:  (1, 256, 256) grayscale xylem slice
      output: 5 physics metrics (Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy)
    """

    def __init__(self, num_outputs: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 16 x 128 x 128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 64 x 32 x 32
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


def load_surrogate_dataset() -> TensorDataset:
    if not os.path.exists(SURROGATE_DATASET_PATH):
        raise FileNotFoundError(
            f"{SURROGATE_DATASET_PATH} not found. "
            f"Run build_surrogate_dataset.py first."
        )

    data = torch.load(SURROGATE_DATASET_PATH, map_location="cpu")
    images = data["images"].float()          # (N, 1, H, W), in [0,1]
    metrics = data["metrics"].float()        # (N, 5)
    return TensorDataset(images, metrics)


def split_dataset(dataset: TensorDataset, val_fraction: float = 0.2):
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    return train_subset, val_subset


def train_surrogate(
    batch_size: int = 16,
    num_epochs: int = 60,
    lr: float = 1e-3,
):
    dataset = load_surrogate_dataset()
    train_set, val_set = split_dataset(dataset, val_fraction=0.2)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = PhysicsSurrogateCNN(num_outputs=5).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")

    print("🧪 Training physics surrogate on", DEVICE)
    print(f"✅ Loaded surrogate dataset:")
    print("   images:", dataset.tensors[0].shape)
    print("   metrics:", dataset.tensors[1].shape)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_train += loss.item() * xb.size(0)
            n_train += xb.size(0)

        loss_train = total_train / max(1, n_train)

        # validation
        model.eval()
        total_val = 0.0
        n_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val += loss.item() * xb.size(0)
                n_val += xb.size(0)

        loss_val = total_val / max(1, n_val)

        tag = ""
        if loss_val < best_val:
            best_val = loss_val
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), OUT_PATH)
            tag = "   ✅ New best val loss, model saved → results/physics_surrogate.pth"

        print(
            f"Epoch {epoch:2d}/{num_epochs:2d} | "
            f"Train MSE: {loss_train:.6f} | "
            f"Val MSE: {loss_val:.6f}{tag}"
        )

    print("🏁 Surrogate training complete. Best val MSE:", best_val)


def main():
    train_surrogate()


if __name__ == "__main__":
    main()
