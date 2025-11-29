"""
morphological_convergence_curve.py
-------------------------------------
Plots the evolution of morphological realism metrics
(latent distance, overlap, SSIM) across training epochs.

Assumes you’ve logged metrics after each epoch in
results/morphology_metrics/history.json
"""

import os
import json
import matplotlib.pyplot as plt

HISTORY_PATH = "results/morphology_metrics/history.json"
SAVE_PATH = "results/morphology_metrics/morphological_convergence_curve.png"

if not os.path.exists(HISTORY_PATH):
    print("⚠️ No history.json found. You need to enable epoch logging in morphology_metrics.py")
    exit()

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = list(range(1, len(history["latent_distance"]) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["latent_distance"], label="Latent Distance ↓", marker="o")
plt.plot(epochs, history["cluster_overlap"], label="Cluster Overlap ↑", marker="s")
plt.plot(epochs, history["ssim"], label="SSIM ↑", marker="^")

plt.title("Morphological Convergence Over Training")
plt.xlabel("Epoch")
plt.ylabel("Metric Value (normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.close()

print(f"✅ Saved morphological convergence curve → {SAVE_PATH}")
