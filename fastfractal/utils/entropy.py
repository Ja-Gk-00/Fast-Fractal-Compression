from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def entropy01(block01: NDArray[np.float32]) -> float:
    x = np.clip(block01 * 255.0, 0.0, 255.0).astype(np.uint8, copy=False).ravel()
    hist = np.bincount(x, minlength=256).astype(np.float64, copy=False)
    p = hist / float(x.size)
    p = p[p > 0.0]
    h = -np.sum(p * np.log2(p))
    return float(h / 8.0)
