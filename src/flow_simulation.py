import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F

# ================================================================
# Flow Simulation Configuration
# ================================================================

RESULTS_DIR = "results/flow_simulation"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Standard target size (consistent across datasets)
TARGET_SIZE = (256, 256)


# ================================================================
# 1. Load and preprocess images (resize + grayscale + normalize)
# ================================================================
def load_grayscale_images(path):
    """
    Loads grayscale images, resizes them to a uniform shape,
    normalizes intensities, enhances contrast, and thresholds
    to separate open vs. solid regions.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚫 Dataset folder not found: {path}")

    files = sorted([f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if len(files) == 0:
        raise RuntimeError(f"No image files found in {path}")

    imgs = []
    for f in files:
        img = Image.open(os.path.join(path, f)).convert("L")

        # ✅ Resize all to uniform target size
        img = img.resize(TARGET_SIZE, Image.BICUBIC)

        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        # Invert if pores are dark
        if arr.mean() < 0.5:
            arr = 1 - arr

        # ✅ Enhance contrast
        arr = np.clip((arr - 0.5) * 2.0 + 0.5, 0, 1)

        # ✅ Binary threshold
        arr = np.where(arr > 0.6, 1.0, 0.0)

        imgs.append(arr)

    return np.stack(imgs, axis=0)


# ================================================================
# 2. Pseudo-physics flow solver
# ================================================================
def simulate_flow(img, iterations=300):
    """
    Compute a pseudo flow pressure field across the xylem structure.
    img: normalized 2D array where 1 = open, 0 = solid
    """
    h, w = img.shape
    mask = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask = (mask > 0.4).float()

    # ✅ Inlet (top = 1), outlet (bottom = 0)
    y = torch.linspace(1.0, 0.0, steps=h).view(1, 1, h, 1)
    pressure = y.clone() * mask + (1 - mask) * 0.0

    # ✅ Relaxation for flow equilibrium
    for _ in range(iterations):
        avg = F.avg_pool2d(pressure, 3, stride=1, padding=1)
        pressure = avg * mask + pressure * (1 - mask)

    # ✅ Pressure gradient → flow rate
    grad_y, grad_x = torch.gradient(pressure[0, 0])
    flow_rate = (grad_y.abs() * mask[0, 0]).mean().item()

    return pressure.squeeze().numpy(), flow_rate


# ================================================================
# 3. Dataset analysis
# ================================================================
def analyze_dataset_flow(name, path):
    imgs = load_grayscale_images(path)
    flow_eff = []

    for i, img in enumerate(imgs):
        pressure_map, flow_rate = simulate_flow(img)
        flow_eff.append(flow_rate)

        # Optional visual save (for inspection)
        plt.imsave(os.path.join(RESULTS_DIR, f"flow_{name}_{i+1}.png"),
                   pressure_map, cmap="viridis")

    mean_eff = np.mean(flow_eff)
    print(f"🌊 Mean relative flow efficiency ({name}): {mean_eff:.4f}")
    return mean_eff


# ================================================================
# 4. Main Entry Point
# ================================================================
def main():
    print("\n🌿 Simulating flow for Real Xylem structures...")
    real_eff = analyze_dataset_flow("Real Xylem", "data/real_xylem")

    print("\n🌿 Simulating flow for Synthetic Xylem structures...")
    synth_eff = analyze_dataset_flow("Synthetic Xylem", "data/generated_microtubes")

    ratio = synth_eff / (real_eff + 1e-8)
    print(f"\n🧩 Synthetic vs Real Flow Ratio: {ratio:.2f}")
    print(f"✅ Flow simulation complete. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
