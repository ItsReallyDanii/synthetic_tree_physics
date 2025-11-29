import os
from PIL import Image
import numpy as np

# Correct raw and processed directories
INPUT_DIR = "data/real_xylem_raw"
OUTPUT_DIR = "data/real_xylem"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Normalize 0–1
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Optional cleanup: binarize slightly to emphasize vessels
    arr[arr < 0.2] = 0.0
    arr[arr >= 0.2] = 1.0

    return Image.fromarray((arr * 255).astype(np.uint8))

def preprocess_all():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")
    
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"📂 Found {len(files)} real xylem images in {INPUT_DIR}")
    
    for fname in files:
        img = preprocess_image(os.path.join(INPUT_DIR, fname))
        img.save(os.path.join(OUTPUT_DIR, fname))
    
    print(f"✅ Preprocessing complete. Normalized images saved → {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_all()
