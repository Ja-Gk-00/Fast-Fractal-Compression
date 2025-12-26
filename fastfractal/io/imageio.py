from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_image(path: Path) -> NDArray[np.float32]:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 2:
        x = arr.astype(np.float32) / 255.0
        return x
    if arr.ndim == 3 and arr.shape[2] >= 3:
        x = arr[:, :, :3].astype(np.float32) / 255.0
        return x
    raise ValueError("unsupported image shape")


def save_image(path: Path, img01: NDArray[np.float32]) -> None:
    x = np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)
    if x.ndim == 2:
        Image.fromarray(x, mode="L").save(path)
        return
    if x.ndim == 3 and x.shape[2] == 3:
        Image.fromarray(x, mode="RGB").save(path)
        return
    raise ValueError("unsupported image shape")
