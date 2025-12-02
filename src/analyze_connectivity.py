import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from skimage import measure

# Configuration
DATA_DIR = "data/generated_microtubes/"
OUTPUT_CSV = "results/connectivity_metrics.csv"
THRESHOLD = 0.80  # Must match your flow solver!

def calculate_connectivity(img_array, threshold):
    # 1. Binarize: Solid is where pixel <= threshold (assuming white=void, black=solid)
    # If your images are (0=black=solid, 1=white=void), then Solid is < Threshold.
    # Adjust logic if your images are inverted. Assuming standard Xylem: Dark = Wall.
    
    # Normalize to 0-1 if 0-255
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
        
    # Solid Mask (The Material)
    # If pixels are Void-probability (1.0 = hole), then Solid is < 0.8
    solid_mask = img_array < threshold
    
    if not np.any(solid_mask):
        return 0.0, 0.0 # Empty image
        
    # 2. Label Connected Components
    # connectivity=2 means including diagonals
    labels = measure.label(solid_mask, connectivity=2)
    
    if labels.max() == 0:
        return 0.0, 0.0
        
    # 3. Find Largest Component
    regions = measure.regionprops(labels)
    # Sort by area (number of pixels)
    regions.sort(key=lambda x: x.area, reverse=True)
    
    largest_area = regions[0].area
    total_solid_area = np.sum(solid_mask)
    
    # 4. Connectivity Score: (Largest Connected Mass) / (Total Mass)
    # 1.0 = Perfect Monolith. 0.1 = Dust Cloud.
    connectivity_score = largest_area / total_solid_area
    
    return connectivity_score, total_solid_area

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    records = []
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])
    
    print(f"🔍 Analyzing connectivity for {len(files)} generated structures...")
    print(f"   Threshold: {THRESHOLD} (Pixels < {THRESHOLD} are SOLID)")

    for f in files:
        path = os.path.join(DATA_DIR, f)
        try:
            img = Image.open(path).convert('L')
            img_arr = np.array(img)
            
            score, mass = calculate_connectivity(img_arr, THRESHOLD)
            
            records.append({
                "filename": f,
                "connectivity_score": score,
                "solid_pixels": mass
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Save Results
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Summary
    mean_score = df['connectivity_score'].mean()
    print(f"\n✅ Connectivity analysis complete.")
    print(f"   Average Connectivity Score: {mean_score:.4f}")
    print(f"   Results saved to: {OUTPUT_CSV}")
    
    # Check for "Dust"
    dusty_samples = df[df['connectivity_score'] < 0.90]
    if len(dusty_samples) > 0:
        print(f"⚠️  WARNING: {len(dusty_samples)} images have < 90% connectivity (Dust risk).")
        print(dusty_samples.head())
    else:
        print("🎉 SUCCESS: All samples are > 90% connected. No dust clouds found.")

if __name__ == "__main__":
    main()