import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# Flow Simulation Utilities
# ============================================================

def normalize_image(img):
    """Normalize image to [0,1] and ensure float32 dtype."""
    img = np.array(img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def simulate_pressure_field(img, iterations=150):
    """
    Simple iterative diffusion to approximate pressure/flow field.
    img: 2D numpy array in [0,1], where 1=solid, 0=open pore
    """
    h, w = img.shape
    pressure = torch.zeros((1, 1, h, w), dtype=torch.float32)
    mask = torch.tensor(1.0 - img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    for _ in range(iterations):
        pressure = F.avg_pool2d(pressure, 3, stride=1, padding=1)
        pressure *= mask

    grad_y, grad_x = torch.gradient(pressure[0, 0])
    grad_mag = (grad_x.abs() + grad_y.abs()).mean().item()
    return pressure.squeeze().numpy(), grad_mag


def compute_flow_metrics(img):
    """
    Compute basic flow-related metrics from a binary or grayscale pore structure.
    Returns dict with keys: 'K', 'porosity', 'flow_grad', 'anisotropy'
    """
    # --- normalize input ---
    img = normalize_image(img)

    # --- physical proxies ---
    porosity = float(np.mean(img < 0.6))  # open fraction

    # simulate approximate flow field
    pressure_field, grad_mag = simulate_pressure_field(img)

    # permeability proxy (lower gradient magnitude + higher porosity → higher K)
    K = 1.0 / (1.0 + grad_mag * (1.0 - porosity + 1e-8))

    # anisotropy measure: directional gradient ratio
    gy, gx = np.gradient(pressure_field)
    anisotropy = np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + 1e-8)

    return {
        "K": float(K),
        "porosity": float(porosity),
        "flow_grad": float(grad_mag),
        "anisotropy": float(anisotropy),
    }


# ============================================================
# Batch utilities for physics-informed training
# ============================================================

def compute_batch_flow_metrics(batch):
    """
    Computes average K and porosity across a batch of images (PyTorch tensor).
    Used inside physics-informed fine-tuning.
    """
    K_vals, P_vals = [], []
    for img in batch:
        img_np = img.squeeze().detach().cpu().numpy().astype(np.float32)
        metrics = compute_flow_metrics(img_np)
        K_vals.append(metrics["K"])
        P_vals.append(metrics["porosity"])
    return float(np.mean(K_vals)), float(np.mean(P_vals))
