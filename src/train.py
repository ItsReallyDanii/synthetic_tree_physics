import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from src.model import Autoencoder

DATA_DIR = "data/real_xylem"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(dataloader):.6f}")

torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model_base.pth"))
print(f"✅ Training complete. Model saved → {RESULTS_DIR}/model_base.pth")
