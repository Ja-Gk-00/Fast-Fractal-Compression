from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import NDArray


# Transform set helper:
#  0–7   : canonical D4 isometries
#  8–11  : transpose-based isometries
#  12–14 : horizontal shear (experimental)
TransformId = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def shear_x(block: NDArray[np.float32], alpha: float) -> NDArray[np.float32]:
    if abs(alpha) > 0.5:
        raise ValueError("shear factor alpha too large")

    h, w = block.shape
    out = np.zeros_like(block)

    for y in range(h):
        shift = int(round(alpha * y))

        if shift >= 0:
            out[y, shift:w] = block[y, : w - shift]
        else:
            out[y, : w + shift] = block[y, -shift:w]

    return out


def apply_transform_2d(
    block: NDArray[np.float32], t: TransformId
) -> NDArray[np.float32]:
    match t:
        case 0:
            return block
        case 1:
            return np.rot90(block, 1)
        case 2:
            return np.rot90(block, 2)
        case 3:
            return np.rot90(block, 3)
        case 4:
            return np.fliplr(block)
        case 5:
            return np.rot90(np.fliplr(block), 1)
        case 6:
            return np.rot90(np.fliplr(block), 2)
        case 7:
            return np.rot90(np.fliplr(block), 3)
        case 8:
            return block.T
        case 9:
            return np.rot90(block.T, 1)
        case 10:
            return np.rot90(block.T, 2)
        case 11:
            return np.rot90(block.T, 3)
        case 12:  # mild left shear
            return shear_x(block, alpha=-0.25)
        case 13:  # no shear (identity in shear transformations family)
            return shear_x(block, alpha=0.0)
        case 14:  # mild right shear
            return shear_x(block, alpha=0.25)
        case _:
            raise ValueError(f"invalid transform id: {t}")
