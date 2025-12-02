import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import XylemAutoencoder

# CONFIG
MODEL_PATH = "results/model_physics_tuned.pth"
OUTPUT_DIR = "results/gradient_beam/"
BEAM_LENGTH_SLICES = 10
LATENT_DIM = 32

def interpolate_latents(model, n_steps=10):
    device = next(model.parameters()).device
    
    # 1. Generate random batch to find endpoints
    print("🔍 Hunting for Dense and Porous endpoints...")
    z_batch = torch.randn(100, LATENT_DIM).to(device)
    with torch.no_grad():
        recon_batch = model.decode(z_batch)
    
    print(f"   Raw Output Range: [{recon_batch.min():.3f}, {recon_batch.max():.3f}]")
    
    # 2. Dynamic Normalization (The Fix)
    # Stretch the gray output (0.15-0.50) to full range (0.0-1.0)
    batch_min = recon_batch.min()
    batch_max = recon_batch.max()
    recon_norm = (recon_batch - batch_min) / (batch_max - batch_min + 1e-6)
    
    # 3. Relative Porosity
    # Any pixel in the top 50% brightness counts as a "void" for this metric
    porosities = (recon_norm > 0.5).float().mean(dim=[1, 2, 3])
    
    idx_dense = torch.argmin(porosities)
    idx_porous = torch.argmax(porosities)
    
    p_min = porosities[idx_dense].item()
    p_max = porosities[idx_porous].item()
    
    print(f"   Selected Dense: {p_min*100:.1f}% (Relative Porosity)")
    print(f"   Selected Porous: {p_max*100:.1f}% (Relative Porosity)")
    
    z_dense = z_batch[idx_dense]
    z_porous = z_batch[idx_porous]
    
    # 4. Interpolate
    alphas = np.linspace(0, 1, n_steps)
    z_steps = [((1 - a) * z_dense + a * z_porous) for a in alphas]
    return torch.stack(z_steps)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Robust Model Loading
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = XylemAutoencoder().to(device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    model.eval()
    
    # Generate Gradient
    z_gradient = interpolate_latents(model, n_steps=BEAM_LENGTH_SLICES)
    
    print("🌊 Decoding gradient beam...")
    with torch.no_grad():
        beam_slices = model.decode(z_gradient)
        
    # --- VISUALIZATION PROCESSING ---
    beam_slices = beam_slices.cpu().numpy().squeeze()
    
    # 1. Stitch slices into one long beam
    full_beam = np.concatenate([s for s in beam_slices], axis=1)
    
    # 2. Normalize the entire beam for plotting (Stretch contrast to 0-1)
    full_beam_norm = (full_beam - full_beam.min()) / (full_beam.max() - full_beam.min() + 1e-6)
    
    # 3. Calculate Stiffness Profile
    # Inverted: Bright (1.0) is Void, Dark (0.0) is Solid
    # Density = 1.0 - Average Brightness
    beam_density_profile = 1.0 - np.mean(full_beam_norm, axis=0)
    
    # Stiffness Heuristic: E ~ Density^2
    beam_stiffness_profile = beam_density_profile ** 2
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    # Top Plot: Stiffness Curve
    axes[0].plot(beam_stiffness_profile, color='#d62728', linewidth=3)
    axes[0].set_title("Predicted Stiffness Gradient ($E \\propto \\rho^2$)", fontsize=16, fontweight='bold')
    axes[0].set_ylabel("Stiffness Potential", fontsize=12)
    axes[0].set_xlabel("Longitudinal Position", fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Bottom Plot: The Gradient Structure
    # Use 'gray_r' so dense material looks black/dark
    axes[1].imshow(full_beam_norm, cmap='gray_r', aspect='auto')
    axes[1].set_title(f"Generative Functionally Graded Beam ({BEAM_LENGTH_SLICES} Segments)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "gradient_beam_analysis.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ Gradient analysis saved to {save_path}")

if __name__ == "__main__":
    main()