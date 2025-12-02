import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.model import XylemAutoencoder
from src.train_surrogate import PhysicsSurrogateCNN

# CONFIG
MODEL_PATH = "results/model_physics_tuned.pth"
SURROGATE_PATH = "results/physics_surrogate.pth"
OUTPUT_DIR = "results/inverse_design/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Differentiable Heuristics
# ---------------------------------------------------------

def calc_stiffness_proxy(image):
    """
    Heuristic: E ~ rho^2. 
    Density (rho) = 1.0 - mean_pixel_value (assuming 0=Solid, 1=Void in normalized space)
    """
    # Normalize image to 0-1 range just in case
    img_min, img_max = image.min(), image.max()
    norm_img = (image - img_min) / (img_max - img_min + 1e-6)
    
    # Calculate relative porosity (bright pixels)
    # Density is the inverse of porosity
    density = 1.0 - torch.mean(norm_img, dim=[1, 2, 3])
    return density ** 2

def total_variation_loss(img):
    """
    Differentiable proxy for 'connectivity'.
    Penalizes noise and disconnected pixel dust.
    """
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (b * c * h * w)

# ---------------------------------------------------------
# Optimization Loop
# ---------------------------------------------------------

def inverse_design(ae, surrogate, target_flow, target_stiffness, steps=200):
    device = next(ae.parameters()).device
    
    # 1. Initialize Z (Random Gaussian)
    latent_dim = 32
    z = torch.randn(1, latent_dim, device=device, requires_grad=True)
    
    # 2. Optimizer
    optimizer = optim.Adam([z], lr=0.05)
    
    # Weights for the Multi-Objective Optimization
    w_flow = 20.0      # Priority 1: Match Flow
    w_stiff = 10.0     # Priority 2: Match Stiffness
    w_tv = 0.1         # Priority 3: Keep it clean (connected)
    
    history = []

    print(f"🎯 Target: Flow={target_flow:.4f}, Stiffness={target_stiffness:.4f}")

    for i in range(steps):
        optimizer.zero_grad()
        
        # Forward Pass
        recon = ae.decode(z)
        
        # Physics Prediction (Surrogate)
        # Surrogate outputs: [Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy]
        # We want to match FlowRate (index 2)
        surrogate_preds = surrogate(recon)
        current_flow = surrogate_preds[:, 2] 
        
        # Stiffness Calculation
        current_stiffness = calc_stiffness_proxy(recon)
        
        # Loss Calculation
        loss_flow = (current_flow - target_flow) ** 2
        loss_stiff = (current_stiffness - target_stiffness) ** 2
        loss_tv = total_variation_loss(recon)
        
        total_loss = (w_flow * loss_flow) + (w_stiff * loss_stiff) + (w_tv * loss_tv)
        
        total_loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"   Step {i:03d}: Flow={current_flow.item():.4f}, Stiff={current_stiffness.item():.4f} | Loss={total_loss.item():.4f}")
            
        history.append({
            'step': i,
            'flow': current_flow.item(),
            'stiffness': current_stiffness.item(),
            'loss': total_loss.item()
        })

    return z.detach(), recon.detach(), pd.DataFrame(history)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Autoencoder
    print("⏳ Loading Autoencoder...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    ae = XylemAutoencoder().to(device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        ae.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        ae.load_state_dict(checkpoint)
    else:
        ae.load_state_dict(checkpoint.state_dict())
    ae.eval()
    
    # Load Surrogate
    print("⏳ Loading Physics Surrogate...")
    surrogate = PhysicsSurrogateCNN().to(device)
    surrogate_ckpt = torch.load(SURROGATE_PATH, map_location=device, weights_only=False)
    if isinstance(surrogate_ckpt, dict):
        surrogate.load_state_dict(surrogate_ckpt)
    else:
        surrogate.load_state_dict(surrogate_ckpt.state_dict())
    surrogate.eval()
    
    # --- DESIGN SWEEP ---
    # These are the specific engineering specs we want to invent
    targets = [
        {"name": "HighFlow_Flexible", "f": 0.0010, "s": 0.2}, # Like a sponge
        {"name": "Balanced_Hybrid",   "f": 0.0008, "s": 0.5}, # Like real xylem
        {"name": "LowFlow_Stiff",     "f": 0.0004, "s": 0.8}, # Like a brick
    ]
    
    for t in targets:
        print(f"\n🚀 Inverse Designing: {t['name']}")
        z_opt, img_opt, hist = inverse_design(ae, surrogate, t['f'], t['s'])
        
        # Save Structure
        img_np = img_opt.cpu().squeeze().numpy()
        plt.imsave(os.path.join(OUTPUT_DIR, f"{t['name']}_structure.png"), img_np, cmap='gray_r')
        
        # Save Convergence Plot
        plt.figure(figsize=(10,4))
        plt.plot(hist['step'], hist['flow'], label='Flow Rate', color='blue')
        plt.plot(hist['step'], hist['stiffness'], label='Stiffness', color='orange')
        plt.axhline(y=t['f'], color='blue', linestyle='--', alpha=0.5, label='Target Flow')
        plt.axhline(y=t['s'], color='orange', linestyle='--', alpha=0.5, label='Target Stiff')
        plt.legend()
        plt.title(f"Optimization Trajectory: {t['name']}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{t['name']}_trajectory.png"))
        plt.close()

    print("\n✅ Design Sweep Complete. Check 'results/inverse_design/' for your artifacts.")

if __name__ == "__main__":
    main()