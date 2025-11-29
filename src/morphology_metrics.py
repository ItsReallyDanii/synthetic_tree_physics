"""
morphology_metrics.py
-------------------------------------
Quantifies morphological similarity between:
  - Real xylem images (UruDendro dataset)
  - Synthetic microtube structures
  - Post-hybrid model reconstructions

Outputs latent distance, SSIM, and cluster overlap indices.
Now logs history across runs for morphological convergence visualization.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.model import XylemAutoencoder
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ---------------------------------------
# CONFIG
# ---------------------------------------
REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"
MODEL_PATH_BASE = "results/model.pth"
MODEL_PATH_HYBRID = "results/hybrid_training/model_hybrid.pth"
SAVE_DIR = "results/morphology_metrics"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
LATENT_DIM = 32

# ---------------------------------------
# DATASET
# ---------------------------------------
class XylemDataset(Dataset):
    def __init__(self, path, transform=None):
        self.files = [os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------------------------------------
# UTILITIES
# ---------------------------------------
def extract_latents(model, dataloader):
    model.eval()
    latents, recons, imgs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            recon, z = model(batch)
            latents.append(z.cpu().numpy())
            recons.append(recon.cpu().numpy())
            imgs.append(batch.cpu().numpy())
    return np.concatenate(latents), np.concatenate(recons), np.concatenate(imgs)

def latent_distance(z_real, z_syn):
    """Compute mean latent L2 distance between real and synthetic embeddings."""
    dist = pairwise_distances(z_real, z_syn, metric="euclidean")
    return np.mean(dist)

def cluster_overlap_index(z_real, z_syn):
    """Rough cluster overlap using t-SNE and centroid distance ratio."""
    pca = PCA(n_components=10).fit_transform(np.concatenate([z_real, z_syn]))
    tsne = TSNE(n_components=2, perplexity=10, learning_rate="auto").fit_transform(pca)
    real_pts, syn_pts = tsne[:len(z_real)], tsne[len(z_real):]
    centroid_dist = np.linalg.norm(real_pts.mean(0) - syn_pts.mean(0))
    cluster_spread = np.std(tsne, axis=0).mean()
    return 1 - min(1.0, centroid_dist / (2 * cluster_spread))

def compute_ssim(real_recon, syn_recon):
    """Compute structural similarity index between average reconstructions."""
    real_mean = np.mean(real_recon, axis=0).squeeze()
    syn_mean = np.mean(syn_recon, axis=0).squeeze()
    return ssim(real_mean, syn_mean, data_range=1.0)

# ---------------------------------------
# JSON SAVE FUNCTION
# ---------------------------------------
def save_metrics_to_json(metrics_base, metrics_hybrid, save_dir=SAVE_DIR):
    summary = {}
    for key in metrics_base:
        summary[key] = {
            "pre": float(metrics_base[key]),
            "post": float(metrics_hybrid[key]),
            "improvement_%": float(
                ((metrics_base[key] - metrics_hybrid[key]) / metrics_base[key]) * 100
            ) if metrics_base[key] != 0 else 0
        }

    json_path = os.path.join(save_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"💾 Saved metrics JSON to {json_path}")

# ---------------------------------------
# MAIN
# ---------------------------------------
def analyze_model(model_path, label):
    print(f"🧠 Evaluating model: {label}")
    model = XylemAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_dict = model.state_dict()
    compat = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compat)
    model.load_state_dict(model_dict)
    print(f"✅ Loaded {len(compat)} layers from {model_path}")

    # Load data
    real_loader = DataLoader(XylemDataset(REAL_DIR, transform), batch_size=BATCH_SIZE)
    syn_loader = DataLoader(XylemDataset(SYN_DIR, transform), batch_size=BATCH_SIZE)

    z_real, recon_real, imgs_real = extract_latents(model, real_loader)
    z_syn, recon_syn, imgs_syn = extract_latents(model, syn_loader)

    metrics = {
        "latent_distance": latent_distance(z_real, z_syn),
        "cluster_overlap": cluster_overlap_index(z_real, z_syn),
        "ssim": compute_ssim(recon_real, recon_syn)
    }

    print(f"\n📊 {label} Morphology Metrics:")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")
    print()

    # Save t-SNE plot
    pca = PCA(n_components=10).fit_transform(np.concatenate([z_real, z_syn]))
    tsne = TSNE(n_components=2, perplexity=10, learning_rate="auto").fit_transform(pca)
    plt.figure(figsize=(6, 5))
    plt.scatter(tsne[:len(z_real), 0], tsne[:len(z_real), 1], label="Real", alpha=0.7)
    plt.scatter(tsne[len(z_real):, 0], tsne[len(z_real):, 1], label="Synthetic", alpha=0.7)
    plt.title(f"Latent Overlap ({label})")
    plt.legend()
    plot_path = os.path.join(SAVE_DIR, f"latent_overlap_{label}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📈 Saved plot: {plot_path}")

    # ---------------------------------------
    # 🧩 Log metrics history for convergence tracking
    # ---------------------------------------
    history_path = os.path.join(SAVE_DIR, "history.json")
    if not os.path.exists(history_path):
        history = {"latent_distance": [], "cluster_overlap": [], "ssim": []}
    else:
        with open(history_path, "r") as f:
            history = json.load(f)

    for key in metrics:
        history[key].append(float(metrics[key]))

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"🧾 Updated history log → {history_path}")

    return metrics

# ---------------------------------------
# RUN
# ---------------------------------------
if __name__ == "__main__":
    metrics_base = analyze_model(MODEL_PATH_BASE, "Pre-Hybrid")
    metrics_hybrid = analyze_model(MODEL_PATH_HYBRID, "Post-Hybrid")

    print("\n🔬 Morphological Convergence Summary")
    print("=" * 50)
    for key in metrics_base:
        base, hybrid = metrics_base[key], metrics_hybrid[key]
        delta = ((base - hybrid) / base) * 100 if base != 0 else 0
        trend = "↑ improved" if hybrid < base else "↓ worse" if hybrid > base else "– no change"
        print(f"{key:20s}: {base:.4f} → {hybrid:.4f} ({trend}, Δ{abs(delta):.1f}%)")

    # Save metrics JSON for dashboard
    save_metrics_to_json(metrics_base, metrics_hybrid)
