"""
Phase 2: Cloud Simulation module.
Generates realistic multi-scale synthetic cloud masks and simulated cloudy images
across coverage bins (10-30%, 30-60%, 60-80%) for thick opaque and semi-transparent conditions.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class CloudSimulator:
    def __init__(self, height=64, width=64):
        self.height = height
        self.width = width

    def generate_cloud_mask(self, target_bin=(0.30, 0.60), rng=None):
        """
        Generates a realistic organic cloud mask within a given coverage percentage bin [low, high].
        Uses multi-scale gaussian filtered noise.
        """
        if rng is None:
            rng = np.random.default_rng()
            
        low, high = target_bin
        target_cov = rng.uniform(low, high)
        
        # Combine multiple octaves of Gaussian smoothed noise for realistic cloud clumps
        noise1 = rng.standard_normal((self.height, self.width))
        noise2 = rng.standard_normal((self.height, self.width))
        noise3 = rng.standard_normal((self.height, self.width))
        
        smooth1 = gaussian_filter(noise1, sigma=rng.uniform(6.0, 12.0))
        smooth2 = gaussian_filter(noise2, sigma=rng.uniform(3.0, 6.0))
        smooth3 = gaussian_filter(noise3, sigma=rng.uniform(1.0, 2.5))
        
        combined = smooth1 * 0.6 + smooth2 * 0.3 + smooth3 * 0.1
        
        # Add random gradient across image to simulate directional cloud drift
        gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        combined += 0.3 * (gx * xx + gy * yy)
        
        # Threshold at percentile to get exact desired coverage
        threshold = np.percentile(combined, 100.0 * (1.0 - target_cov))
        mask = (combined >= threshold).astype(np.uint8)
        
        return mask

    def generate_shape_mask(self, target_bin=(0.30, 0.60), rng=None):
        """
        Held-out mask family (random ellipses + rectangles), used ONLY to train the
        adapted classifiers (mask-aware, fill-aware). Test clouds come from
        generate_cloud_mask, so adapted classifiers never see the test mask distribution.
        Coverage lands inside target_bin (shapes are added until it is reached).
        """
        if rng is None:
            rng = np.random.default_rng()
        low, high = target_bin
        target_cov = rng.uniform(low, high)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        total = float(mask.size)
        for _ in range(200):
            cx, cy = int(rng.integers(0, self.width)), int(rng.integers(0, self.height))
            ax, ay = int(rng.integers(6, 28)), int(rng.integers(6, 28))
            trial = mask.copy()
            if rng.random() < 0.5:
                cv2.ellipse(trial, (cx, cy), (ax, ay), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            else:
                cv2.rectangle(trial, (cx - ax, cy - ay), (cx + ax, cy + ay), 1, -1)
            cov = trial.sum() / total
            if cov <= high:
                mask = trial
            if mask.sum() / total >= target_cov:
                break
        return mask

    def apply_cloud(self, clean_img, mask, condition="thick", rng=None):
        """
        clean_img: uint8 HxWx3 RGB
        mask: uint8 HxW (1 where cloud is present, 0 where clear)
        condition: 'thick' (opaque) or 'semi_transparent' (50% alpha)
        
        Returns:
            cloudy_img: uint8 HxWx3 RGB
            descriptors: dict with coverage_pct, cloud_brightness, thickness_proxy
        """
        if rng is None:
            rng = np.random.default_rng()
            
        clean_f = clean_img.astype(np.float32)
        H, W, C = clean_img.shape
        
        # Base cloud color: white/light grey with subtle texture variance
        base_cloud = rng.normal(loc=238.0, scale=8.0, size=(H, W, 1)).astype(np.float32)
        base_cloud = np.clip(base_cloud, 210.0, 255.0)
        base_cloud = np.repeat(base_cloud, 3, axis=-1)
        
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        
        if condition == "thick":
            # Opaque cloud: completely obscures the ground surface
            cloudy_f = clean_f * (1.0 - mask_3d) + base_cloud * mask_3d
        elif condition == "semi_transparent":
            # 50% opacity: partially transmits surface signal
            alpha = 0.50
            blended = alpha * base_cloud + (1.0 - alpha) * clean_f
            cloudy_f = clean_f * (1.0 - mask_3d) + blended * mask_3d
        else:
            raise ValueError(f"Unknown condition: {condition}")
            
        cloudy_img = np.clip(cloudy_f, 0, 255).astype(np.uint8)
        
        # Compute descriptors
        cloud_pixels = mask.sum()
        total_pixels = mask.size
        coverage_pct = float(cloud_pixels / total_pixels)
        
        if cloud_pixels > 0:
            cloud_brightness = float(cloudy_img[mask == 1].mean())
            thickness_proxy = float(np.abs(cloudy_f[mask_3d == 1] - clean_f[mask_3d == 1]).mean())
        else:
            cloud_brightness = 0.0
            thickness_proxy = 0.0
            
        descriptors = {
            "coverage_pct": coverage_pct,
            "cloud_brightness": cloud_brightness,
            "thickness_proxy": thickness_proxy,
            "condition": condition
        }
        
        return cloudy_img, descriptors

if __name__ == "__main__":
    sim = CloudSimulator(64, 64)
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8) + 100
    
    for b_name, b_range in [("10-30%", (0.10, 0.30)), ("30-60%", (0.30, 0.60)), ("60-80%", (0.60, 0.80))]:
        m = sim.generate_cloud_mask(b_range)
        cloudy, desc = sim.apply_cloud(dummy_img, m, condition="thick")
        print(f"Bin {b_name}: Coverage={desc['coverage_pct']:.3f}, Brightness={desc['cloud_brightness']:.1f}, ThickProxy={desc['thickness_proxy']:.1f}")
        
    cloudy_semi, desc_semi = sim.apply_cloud(dummy_img, m, condition="semi_transparent")
    print(f"Semi-transparent: Coverage={desc_semi['coverage_pct']:.3f}, Brightness={desc_semi['cloud_brightness']:.1f}, ThickProxy={desc_semi['thickness_proxy']:.1f}")
