import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
import torch


REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"
METRICS_CSV = "results/flow_metrics/flow_metrics.csv"
OUT_PATH = "results/surrogate_dataset.pt"

METRIC_COLS = ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"]


def list_pngs(root: str) -> List[str]:
    """Return sorted list of PNG paths inside a directory."""
    files = [f for f in os.listdir(root) if f.lower().endswith(".png")]
    files.sort()
    return [os.path.join(root, f) for f in files]


def load_image(path: str) -> torch.Tensor:
    """Load 256x256 grayscale image and normalize to [0, 1]."""
    img = Image.open(path).convert("L")
    # Your preprocessed images should already be 256×256, but this is cheap insurance.
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)     # (1, H, W)
    return tensor


def build_surrogate_dataset() -> None:
    # ----- 1. Load solver metrics -----
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"Missing {METRICS_CSV}. Run flow_simulation.py first.")

    df = pd.read_csv(METRICS_CSV)

    for col in METRIC_COLS + ["Type"]:
        if col not in df.columns:
            raise RuntimeError(f"Column '{col}' not found in {METRICS_CSV}")

    df_real = df[df["Type"] == "real"].reset_index(drop=True)
    df_syn = df[df["Type"] == "synthetic"].reset_index(drop=True)

    # ----- 2. Collect images in the SAME order as flow_simulation.py -----
    real_paths = list_pngs(REAL_DIR)
    syn_paths = list_pngs(SYN_DIR)

    if len(real_paths) != len(df_real):
        print(f"⚠️ real image count ({len(real_paths)}) != real metrics rows ({len(df_real)})")
    if len(syn_paths) != len(df_syn):
        print(f"⚠️ synthetic image count ({len(syn_paths)}) != synthetic metrics rows ({len(df_syn)})")

    all_images = []
    all_metrics = []

    # Real first, then synthetic — this matches how flow_simulation writes the CSV
    for img_path, (_, row) in zip(real_paths, df_real[METRIC_COLS].iterrows()):
        img_t = load_image(img_path)
        all_images.append(img_t)
        all_metrics.append(torch.from_numpy(row.values.astype(np.float32)))

    for img_path, (_, row) in zip(syn_paths, df_syn[METRIC_COLS].iterrows()):
        img_t = load_image(img_path)
        all_images.append(img_t)
        all_metrics.append(torch.from_numpy(row.values.astype(np.float32)))

    if not all_images:
        raise RuntimeError("No images collected; cannot build surrogate dataset")

    images_tensor = torch.stack(all_images, dim=0)   # (N, 1, H, W)
    metrics_tensor = torch.stack(all_metrics, dim=0) # (N, 5)

    # ----- 3. Save dataset -----
    torch.save(
        {
            "images": images_tensor,
            "metrics": metrics_tensor,
            "metric_names": METRIC_COLS,
        },
        OUT_PATH,
    )

    print("✅ Built surrogate dataset")
    print("   images: ", images_tensor.shape)
    print("   metrics:", metrics_tensor.shape)
    print("   saved → ", OUT_PATH)


if __name__ == "__main__":
    build_surrogate_dataset()
