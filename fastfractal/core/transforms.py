from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TransformId = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


@lru_cache(maxsize=64)
def _shear_shifts(h: int, alpha_q: int) -> tuple[int, ...]:
    alpha = alpha_q / 1024.0
    return tuple(int(round(alpha * y)) for y in range(h))


def shear_x(block: NDArray[np.float32], alpha: float) -> NDArray[np.float32]:
    if alpha == 0.0:
        return block
    if abs(alpha) > 0.5:
        raise ValueError("shear factor alpha too large")

    h, w = block.shape
    out = np.zeros_like(block)
    alpha_q = int(round(float(alpha) * 1024.0))
    shifts = _shear_shifts(h, alpha_q)

    for y, shift in enumerate(shifts):
        if shift == 0:
            out[y, :] = block[y, :]
        elif shift > 0:
            if shift < w:
                out[y, shift:w] = block[y, : w - shift]
        else:
            s = -shift
            if s < w:
                out[y, : w - s] = block[y, s:w]

    return out


def apply_transform_2d(
    block: NDArray[np.float32], t: TransformId | int
) -> NDArray[np.float32]:
    tt = int(t)
    b = block

    # 0–7: canonical D4 isometries
    if tt == 0:
        return b
    if tt == 1:
        return b[::-1, :].T
    if tt == 2:
        return b[::-1, ::-1]
    if tt == 3:
        return b.T[::-1, :]
    if tt == 4:
        return b[:, ::-1]
    if tt == 5:
        return b.T[::-1, ::-1]
    if tt == 6:
        return b[::-1, :]
    if tt == 7:
        return b.T

    # 8–11: transpose-based isometries
    if tt == 8:
        return b.T
    if tt == 9:
        return b[:, ::-1]
    if tt == 10:
        return b.T[::-1, ::-1]
    if tt == 11:
        return b[::-1, :]

    # 12–14: shear transformations
    if tt == 12:
        return shear_x(b, alpha=-0.25)
    if tt == 13:
        return shear_x(b, alpha=0.0)
    if tt == 14:
        return shear_x(b, alpha=0.25)

    raise ValueError(f"invalid transform id: {t}")
