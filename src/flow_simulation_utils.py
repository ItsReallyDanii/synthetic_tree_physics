import numpy as np
import torch
import torch.nn.functional as F

# ==================================================
# 💧 Compute Flow Physics Metrics
# ==================================================
def compute_flow_metrics(img, grad_scale=1.0):
    """
    Compute approximate flow physics metrics for a 2D xylem microstructure image.

    Args:
        img (np.ndarray): Grayscale image, either in [0, 1] or [0, 255].
        grad_scale (float): Optional multiplier to amplify flow sensitivity.

    Returns:
        dict with keys:
            - "K":         effective permeability proxy
            - "Porosity":  fraction of open (bright) pixels
            - "Anisotropy":ratio of |∂x| to |∂y|
    """
    ...
    # img is assumed to have been converted to a torch tensor called `tensor`
    # with shape [1, 1, H, W], and we already computed gx, gy, grad_mag above.

    # --------------------------------------------------
    # ✅ FIXED POROSITY DEFINITION (Step #3)
    # --------------------------------------------------
    # We first make sure the image is in [0, 1]. If it’s still 0–255,
    # we normalize here. Then we apply a high threshold (0.80) so that
    # only very bright voxels count as “void” / open space.
    max_val = float(tensor.max().item())
    if max_val > 1.5:
        # Likely 0–255 input → normalize
        norm = tensor / 255.0
    else:
        # Already 0–1
        norm = tensor

    # Bright > 0.80 = void; rest = solid
    binary = (norm > 0.80).float()
    porosity = binary.mean().item()

    # Effective permeability proxy (inverse relation with gradient magnitude)
    K = (1.0 / (1.0 + grad_mag * (1 - porosity))).mean().item()

    # Directional anisotropy ratio (|∂x| vs |∂y|)
    anisotropy = (gx.abs().mean() / (gy.abs().mean() + 1e-8)).item()

    return {
        "K": K,
        "Porosity": porosity,
        "Anisotropy": anisotropy,
    }
