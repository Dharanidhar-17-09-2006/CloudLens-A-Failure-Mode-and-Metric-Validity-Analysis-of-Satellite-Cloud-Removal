"""Dark channel prior dehazing (He et al.), numpy/OpenCV only. Takes the real cloudy tile, ignores the mask."""
import numpy as np, cv2


def _dark(img, patch):
    return cv2.erode(img.min(axis=2), np.ones((patch, patch), np.uint8))


def dehaze_dcp(rgb, patch=7, omega=0.95, t0=0.1, top=0.01):
    I = rgb.astype(np.float32) / 255.0
    dark = _dark(I, patch)
    n = max(1, int(dark.size * top))
    idx = np.argsort(dark.ravel())[-n:]
    px = I.reshape(-1, 3)[idx]
    A = px[px.sum(1).argmax()].clip(0.3, 1.0)
    t = 1.0 - omega * _dark(I / A, patch)
    t = cv2.GaussianBlur(t, (0, 0), 2).clip(t0, 1.0)[..., None]
    J = (I - A) / t + A
    return (J.clip(0, 1) * 255).astype(np.uint8)


def dehaze_dcp_masked(rgb, mask, **kw):
    """DCP output only inside the cloud mask; clear pixels are left untouched."""
    out = dehaze_dcp(rgb, **kw)
    return np.where(mask[..., None].astype(bool), out, rgb)
