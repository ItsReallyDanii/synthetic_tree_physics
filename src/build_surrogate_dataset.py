"""
build_surrogate_dataset.py
----------------------------------------
Builds a supervised dataset for the physics surrogate directly from
images + solver (compute_flow_metrics), without relying on any CSV.

For each image in:
    - data/real_xylem_preprocessed  (type="Real")
    - data/generated_microtubes     (type="Synthetic")

we compute:
    - Mean_K
    - Mean_dP/dy
    - FlowRate
    - Porosity
    - Anisotropy

and pack them into a single .pt file:

    results/surrogate_dataset.pt
        {
            "images":    FloatTensor [N, 1, 256, 256],
            "metrics":   FloatTensor [N, 5],
            "filenames": list[str],
            "types":     list[str],  # "Real" | "Synthetic"
            "metric_names": ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"],
        }
"""

import os
import numpy as np
import torch
from PIL import Image

from src.flow_simulation_utils import compute_flow_metrics  # same helper used in your pipeline

# Directories
REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"

# Output
OUT_PATH = "results/surrogate_dataset.pt"


def load_image(path, size=(256, 256)):
    """Load grayscale image, resize, normalize to [0,1], return tensor [1,H,W]."""
    img = Image.open(path).convert("L").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


def extract_metrics_from_solver(img_tensor):
    """
    Run compute_flow_metrics on a single image tensor [1,H,W]
    and extract the five core metrics we care about.

    Returns: (Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy)
    """
    img_np = img_tensor.squeeze(0).numpy()  # [H,W]

    raw = compute_flow_metrics(img_np)  # expect dict-like

    # Robust key lookup (case-insensitive, tolerate small naming variations)
    keys_norm = {}
    for k, v in raw.items():
        k_norm = "".join(c for c in k.lower() if c.isalnum())  # e.g. "mean_dp/dy" -> "meandpdy"
        keys_norm[k_norm] = v

    def grab(*candidates, default=0.0):
        for cand in candidates:
            cn = "".join(c for c in cand.lower() if c.isalnum())
            if cn in keys_norm and keys_norm[cn] is not None:
                return float(keys_norm[cn])
        return float(default)

    mean_k     = grab("mean_k", "k")
    mean_dpdy  = grab("mean_dp/dy", "mean_dp_dy", "dpdy")
    flowrate   = grab("flowrate", "flow_rate")
    porosity   = grab("porosity")
    anisotropy = grab("anisotropy")

    return mean_k, mean_dpdy, flowrate, porosity, anisotropy


def collect_from_dir(root_dir, type_label):
    """
    Iterate over all images in a directory, compute metrics via solver,
    and return lists of tensors/values.
    """
    images = []
    metrics = []
    filenames = []
    types = []

    if not os.path.isdir(root_dir):
        print(f"⚠️ Directory not found, skipping: {root_dir}")
        return images, metrics, filenames, types

    files = sorted(
        f for f in os.listdir(root_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    )

    if not files:
        print(f"⚠️ No images found in {root_dir}")
        return images, metrics, filenames, types

    print(f"📂 Collecting from {root_dir} ({type_label}), {len(files)} images...")

    for fname in files:
        path = os.path.join(root_dir, fname)
        try:
            x = load_image(path)  # [1,256,256]
            m = extract_metrics_from_solver(x)  # tuple of 5 floats
        except Exception as e:
            print(f"⚠️ Failed on {path}: {e}")
            continue

        images.append(x)
        metrics.append(torch.tensor(m, dtype=torch.float32))
        filenames.append(fname)
        types.append(type_label)

    return images, metrics, filenames, types


def main():
    all_images = []
    all_metrics = []
    all_filenames = []
    all_types = []

    # Real set
    imgs, mets, fnames, tlabels = collect_from_dir(REAL_DIR, "Real")
    all_images.extend(imgs)
    all_metrics.extend(mets)
    all_filenames.extend(fnames)
    all_types.extend(tlabels)

    # Synthetic set
    imgs, mets, fnames, tlabels = collect_from_dir(SYN_DIR, "Synthetic")
    all_images.extend(imgs)
    all_metrics.extend(mets)
    all_filenames.extend(fnames)
    all_types.extend(tlabels)

    if not all_images:
        raise RuntimeError("No images processed; dataset would be empty.")

    X = torch.stack(all_images, dim=0)   # [N,1,256,256]
    Y = torch.stack(all_metrics, dim=0)  # [N,5]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(
        {
            "images": X,
            "metrics": Y,
            "filenames": all_filenames,
            "types": all_types,
            "metric_names": ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"],
        },
        OUT_PATH,
    )

    print("✅ Built surrogate dataset")
    print(f"   images:  {X.shape}")
    print(f"   metrics: {Y.shape}")
    print(f"   saved →  {OUT_PATH}")


if __name__ == "__main__":
    main()
