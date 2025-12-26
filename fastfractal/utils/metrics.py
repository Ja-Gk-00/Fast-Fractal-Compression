from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mse(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    d = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.mean(d * d))


def psnr(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    v = mse(a, b)
    if v <= 0.0:
        return 99.0
    return float(10.0 * np.log10(1.0 / v))
