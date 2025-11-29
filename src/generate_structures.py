"""
generate_structures.py
Creates synthetic 'xylem-like' microtube cross-sections for Phase 1.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "data/generated_microtubes"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_structure(seed: int = 0, n_tubes: int = 5):
    np.random.seed(seed)
    img = np.zeros((256, 256))
    for _ in range(n_tubes):
        r = np.random.randint(10, 30)
        x, y = np.random.randint(r, 256 - r, 2)
        yy, xx = np.ogrid[:256, :256]
        mask = (xx - x)**2 + (yy - y)**2 <= r**2
        img[mask] = 1
    return img

if __name__ == "__main__":
    for i in range(64):
        arr = generate_structure(i)
        plt.imshow(arr, cmap="gray")
        plt.axis("off")
        plt.savefig(f"{OUT_DIR}/xylem_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
    print(f"✅ Generated 10 structures in {OUT_DIR}")
