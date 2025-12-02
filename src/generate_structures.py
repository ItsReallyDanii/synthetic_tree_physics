import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch import nn
from src.model import XylemAutoencoder
import pandas as pd


def save_image_from_tensor(tensor, path):
    """
    Save a single-channel tensor [1,H,W] or [H,W] as a grayscale PNG using PIL.
    Assumes values are in [0, 1].
    """
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)  # [1,H,W] -> [H,W]
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(path)


def generate_structures(model_path, n=64, out_dir="data/generated_microtubes"):
    """
    Load a tuned autoencoder model and generate N synthetic microtubes
    by sampling latent vectors from a standard normal distribution,
    decoding them to images, and saving them as PNGs.
    Also logs latent vectors for analysis.
    """
    device = torch.device("cpu")

    # Load model
    model = XylemAutoencoder()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    # Latent dimension inferred from model
    latent_dim = model.latent_dim if hasattr(model, "latent_dim") else 32

    generated = []
    with torch.no_grad():
        for i in range(n):
            # Sample a random latent code
            z = torch.randn(1, latent_dim, device=device)

            # Decode to image
            recon = model.decode(z)  # [1,1,256,256]
            img_tensor = recon[0, 0]  # [H,W]

            # Save to disk
            filename = f"synthetic_{i:03d}.png"
            filepath = os.path.join(out_dir, filename)
            save_image_from_tensor(img_tensor, filepath)

            generated.append({
                "filename": filename,
                "z": z.detach().cpu().numpy().flatten().tolist()
            })

    # Save latent log as CSV/JSON-ish
    log_path = os.path.join(out_dir, "generation_log.csv")
    pd.DataFrame(generated).to_csv(log_path, index=False)
    print(f"✅ Generated {n} structures in {out_dir}")
    print(f"🧾 Generation log saved → {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to tuned model (.pth)")
    parser.add_argument("--n", type=int, default=64, help="Number of structures to generate")
    args = parser.parse_args()

    generate_structures(args.model, args.n)


if __name__ == "__main__":
    main()