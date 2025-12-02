import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

RESULTS_DIR = "results/flow_simulation"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------
# KEY CHANGE: porosity "ruler"
# Dark pixels = solid walls, light pixels = void/pores.
# With real images having mean ~0.73, we treat only the
# very bright pixels (> 0.80) as pores.
# -------------------------------------------------------
POROSITY_THRESHOLD = 0.80


def load_grayscale_images(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚫 Dataset folder not found: {path}")

    files = sorted(
        [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    imgs = []
    shapes = set()
    for f in files:
        img = Image.open(os.path.join(path, f)).convert("L")
        arr = np.array(img, dtype=np.float32)
        shapes.add(arr.shape)

        # Normalize to [0,1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        imgs.append(arr)

    if len(shapes) > 1:
        # Resize all to same shape (smallest H,W)
        target_h, target_w = min(s[0] for s in shapes), min(s[1] for s in shapes)
        resized_imgs = []
        for arr in imgs:
            arr = Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (target_w, target_h), Image.BILINEAR
            )
            resized_imgs.append(np.array(arr, dtype=np.float32) / 255.0)
        imgs = resized_imgs

    return np.stack(imgs, axis=0)


def simulate_flow(img, steps=300, diffusivity=0.25):
    """
    Diffusion-based pseudo-flow simulation.

    Assumptions:
      - img is a 2D array in [0,1]
      - Dark pixels (near 0) = solid walls
      - Bright pixels (near 1) = pores / void
    """
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    h, w = img_t.shape[-2:]

    # Pressure initialized as uniform + small perturbation
    pressure = torch.rand_like(img_t) * 0.1

    # Diffusion mask (higher intensity = more flow)
    D = img_t * diffusivity

    for _ in range(steps):
        # 5-point Laplacian kernel
        lap = (
            F.pad(pressure, (0, 0, 1, 1))[..., 2:, :]
            + F.pad(pressure, (0, 0, 1, 1))[..., :-2, :]
            + F.pad(pressure, (1, 1, 0, 0))[..., :, 2:]
            + F.pad(pressure, (1, 1, 0, 0))[..., :, :-2]
            - 4 * pressure
        )
        pressure = pressure + D * lap

    # Compute gradient-based flow rate
    grad_y, grad_x = torch.gradient(pressure[0, 0])
    flow_rate = (grad_x.abs().mean() + grad_y.abs().mean()).item()

    # -------------------------------
    # Physics-ish summary metrics
    # -------------------------------
    mean_k = D.mean().item()
    mean_dpdy = grad_y.abs().mean().item()

    # KEY FIX: porosity uses POROSITY_THRESHOLD instead of 0.5
    # Only pixels brighter than 0.80 count as pores.
    porosity = (img > POROSITY_THRESHOLD).mean().item()

    anisotropy = float(
        grad_x.abs().mean() / (grad_y.abs().mean() + 1e-8)
    )

    return pressure.squeeze().numpy(), {
        "Mean_K": mean_k,
        "Mean_dP/dy": mean_dpdy,
        "FlowRate": flow_rate,
        "Porosity": porosity,
        "Anisotropy": anisotropy,
    }


def analyze_dataset_flow(name, path, tag):
    imgs = load_grayscale_images(path)
    metrics = []
    print(f"\n🌿 Simulating flow for {name} structures...")
    for i, img in enumerate(tqdm(imgs, desc=name)):
        pressure_map, m = simulate_flow(img)
        m["Type"] = tag
        metrics.append(m)
        plt.imsave(
            os.path.join(RESULTS_DIR, f"flow_{name}_{i+1:03d}.png"),
            pressure_map,
            cmap="viridis",
        )

    df = pd.DataFrame(metrics)
    mean_eff = df["FlowRate"].mean()
    print(f"🌊 Mean relative flow efficiency ({name}): {mean_eff:.4f}")
    return df, mean_eff


def main():
    real_df, real_eff = analyze_dataset_flow(
        "Real Xylem",
        "data/real_xylem_preprocessed",  # NEW: use grayscale preprocessed images
        "real",
    )
    synth_df, synth_eff = analyze_dataset_flow(
        "Synthetic Xylem", "data/generated_microtubes", "synthetic"
    )

    df_all = pd.concat([real_df, synth_df], ignore_index=True)
    os.makedirs("results/flow_metrics", exist_ok=True)
    df_all.to_csv("results/flow_metrics/flow_metrics.csv", index=False)
    print("\n💾 Flow metrics saved → results/flow_metrics/flow_metrics.csv")

    ratio = synth_eff / (real_eff + 1e-8)
    print(f"\n🧩 Synthetic vs Real Flow Ratio: {ratio:.2f}")
    print(f"✅ Flow simulation complete. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
