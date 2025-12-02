import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from src.model import XylemAutoencoder
from src.train_surrogate import PhysicsSurrogateCNN

DEVICE = "cpu"
IMG_SIZE = (256, 256)

# Targets from solver stats
TARGET_MEAN_K = 0.247403
TARGET_FLOW = 0.000507
TARGET_POROSITY = 0.990419

N_LATENTS = 8          # number of latent codes to optimize in parallel
N_STEPS = 500          # optimization steps
LR = 1e-2              # learning rate for z
OUT_DIR = "results/optimized_latent_v1"


def save_image_grid(tensor: torch.Tensor, path: str, nrow: int = 4):
    """
    Save a [B,1,H,W] tensor as a simple grayscale grid PNG without torchvision.
    Values are assumed in [0,1].
    """
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    b, c, h, w = tensor.shape
    assert c == 1, "Expected grayscale images [B,1,H,W]"

    nrow = min(nrow, b)
    ncol = math.ceil(b / nrow)

    grid = torch.zeros(1, ncol * h, nrow * w)

    idx = 0
    for row in range(ncol):
        for col in range(nrow):
            if idx >= b:
                break
            grid[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = tensor[idx]
            idx += 1

    img = (grid.squeeze(0).numpy() * 255.0).astype("uint8")
    Image.fromarray(img, mode="L").save(path)


def load_autoencoder(model_path: str) -> XylemAutoencoder:
    model = XylemAutoencoder().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_surrogate(path: str) -> PhysicsSurrogateCNN:
    surrogate = PhysicsSurrogateCNN().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    surrogate.load_state_dict(state)
    surrogate.eval()
    return surrogate


def infer_latent_shape(model: XylemAutoencoder):
    """
    Use a dummy image to see what shape the latent code has.
    Assumes model(x) -> (recon, z).
    """
    dummy = torch.zeros(1, 1, *IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        _, z = model(dummy)
    return z.shape  # (1, latent_dim) or (1,C,H,W)


def physics_loss(pred_metrics: torch.Tensor) -> torch.Tensor:
    """
    pred_metrics shape: [B, 5] = [Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy]
    """
    mean_k = pred_metrics[:, 0]
    flow = pred_metrics[:, 2]
    porosity = pred_metrics[:, 3]

    w_k = 1.0
    w_flow = 0.3
    w_poro = 2.0

    loss_k = (mean_k - TARGET_MEAN_K) ** 2
    loss_poro = (porosity - TARGET_POROSITY) ** 2
    loss_flow = (flow - TARGET_FLOW) ** 2

    return (w_k * loss_k + w_poro * loss_poro + w_flow * loss_flow).mean()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("🌱 Latent optimization with frozen AE + surrogate")
    ae = load_autoencoder("results/model_physics_v04.pth")   # adjust filename if needed
    surrogate = load_surrogate("results/physics_surrogate.pth")

    # Figure out latent shape, create trainable codes
    latent_shape = infer_latent_shape(ae)[1:]  # drop batch dim
    print(f"🧩 Latent shape: {latent_shape}")

    z = torch.randn((N_LATENTS, *latent_shape), device=DEVICE, requires_grad=True)
    optimizer = optim.Adam([z], lr=LR)

    best_loss = float("inf")
    best_images = None
    best_metrics = None

    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()

        # Decode latents -> images
        # Assumes your model has .decode(z), like in generate_structures.py
        imgs = ae.decode(z)                 # [B,1,256,256]
        imgs_clamped = imgs.clamp(0.0, 1.0)

        # Surrogate metrics (in-graph for gradients)
        metrics_pred = surrogate(imgs_clamped)
        loss = physics_loss(metrics_pred)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_images = imgs_clamped.detach().cpu()
                best_metrics = metrics_pred.detach().cpu()

        if step == 1 or step % 25 == 0:
            mk = metrics_pred[:, 0].mean().item()
            fl = metrics_pred[:, 2].mean().item()
            po = metrics_pred[:, 3].mean().item()
            print(
                f"Step {step:4d}/{N_STEPS} | Loss: {loss.item():.6f} | "
                f"Mean_K: {mk:.5f} | Flow: {fl:.5f} | Porosity: {po:.5f}"
            )

    # Save best batch
    if best_images is not None:
        grid_path = os.path.join(OUT_DIR, "optimized_batch.png")
        save_image_grid(best_images, grid_path, nrow=4)
        torch.save(
            {
                "z": z.detach().cpu(),
                "best_images": best_images,
                "best_metrics": best_metrics,
                "best_loss": best_loss,
            },
            os.path.join(OUT_DIR, "optimized_latent_results.pt"),
        )
        print(f"✅ Saved best images + latents → {grid_path}")
        print(f"Best surrogate physics loss: {best_loss:.6f}")
    else:
        print("⚠️ No best_images recorded (this should not happen).")


if __name__ == "__main__":
    main()
