from __future__ import annotations
import numpy as np

def make_uniform_sampler(low: float = -1.0, high: float = 1.0):
    """
    Returns sampler(q, Kx) -> (q, Kx) i.i.d. Uniform[low, high].
    Works for both SoS and SoF (Γ rows are in [-1,1] in your optimizer).
    """
    lo, hi = float(low), float(high)
    def sampler(q: int, kx: int) -> np.ndarray:
        return np.random.uniform(lo, hi, size=(int(q), int(kx))).astype(np.float32)
    return sampler
