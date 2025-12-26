from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

TransformId = Literal[0, 1, 2, 3, 4, 5, 6, 7]


def apply_transform_2d(x: NDArray[np.float32], t: int) -> NDArray[np.float32]:
    if t == 0:
        return x
    if t == 1:
        return np.rot90(x, 1)
    if t == 2:
        return np.rot90(x, 2)
    if t == 3:
        return np.rot90(x, 3)
    if t == 4:
        return np.fliplr(x)
    if t == 5:
        return np.rot90(np.fliplr(x), 1)
    if t == 6:
        return np.rot90(np.fliplr(x), 2)
    if t == 7:
        return np.rot90(np.fliplr(x), 3)
    raise ValueError("invalid transform id")
