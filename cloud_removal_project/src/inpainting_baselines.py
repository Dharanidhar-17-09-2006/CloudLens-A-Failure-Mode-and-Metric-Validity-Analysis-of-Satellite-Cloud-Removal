"""
Phase 3: Classical Inpainting and Simple Fill Baselines.
Implements:
1. cloudy_as_is
2. masked_zero
3. mean_fill
4. nearest_neighbor_fill
5. telea_inpaint
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

def cloudy_as_is(cloudy_img, mask):
    """Returns the cloudy image unchanged."""
    return cloudy_img.copy()

def masked_input(cloudy_img, mask, fill_value=0):
    """Sets cloud-covered pixels to fill_value (0=black)."""
    out = cloudy_img.copy()
    out[mask == 1] = fill_value
    return out

def mean_fill(cloudy_img, mask):
    """
    Computes mean color of unmasked (cloud-free) surface
    and fills the masked region with this average tone.
    """
    out = cloudy_img.copy()
    unmasked = mask == 0
    if unmasked.sum() > 0:
        mean_r = cloudy_img[:, :, 0][unmasked].mean()
        mean_g = cloudy_img[:, :, 1][unmasked].mean()
        mean_b = cloudy_img[:, :, 2][unmasked].mean()
    else:
        mean_r, mean_g, mean_b = 128.0, 128.0, 128.0
        
    out[mask == 1, 0] = np.uint8(np.clip(mean_r, 0, 255))
    out[mask == 1, 1] = np.uint8(np.clip(mean_g, 0, 255))
    out[mask == 1, 2] = np.uint8(np.clip(mean_b, 0, 255))
    return out

def nearest_neighbor_fill(cloudy_img, mask):
    """
    Fills each cloud pixel with the color of the nearest unmasked pixel
    using Euclidean distance transform.
    """
    if mask.sum() == 0:
        return cloudy_img.copy()
    if (mask == 0).sum() == 0:
        return np.zeros_like(cloudy_img)
        
    # distance_transform_edt can return nearest indices
    _, indices = distance_transform_edt(mask, return_indices=True)
    out = np.zeros_like(cloudy_img)
    for c in range(3):
        out[:, :, c] = cloudy_img[:, :, c][indices[0], indices[1]]
    return out

def telea_inpaint(cloudy_img, mask, radius=3):
    """
    Alexandru Telea's Fast Marching Method for image inpainting (OpenCV).
    """
    # OpenCV expects BGR and mask with 255 for inpaint region
    bgr = cv2.cvtColor(cloudy_img, cv2.COLOR_RGB2BGR)
    mask_cv = (mask * 255).astype(np.uint8)
    inpainted_bgr = cv2.inpaint(bgr, mask_cv, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
