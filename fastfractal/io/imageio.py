from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image  # type: ignore[import-not-found, unused-ignore]


def load_image(
    path: Path,
    *,
    mode: Literal["L", "RGB"] | None = None,
    max_side: int | None = 2048,
) -> NDArray[np.float32]:
    with Image.open(path) as im:
        if mode is None:
            mode = "RGB" if len(im.getbands()) >= 3 else "L"
        if max_side is not None and max_side > 0:
            try:
                im.draft(mode, (max_side, max_side))
            except Exception:
                pass

        if im.mode != mode:
            im = im.convert(mode)

        if max_side is not None and max_side > 0:
            w, h = im.size
            if max(w, h) > max_side:
                if w >= h:
                    nw = max_side
                    nh = max(1, int(round(h * (max_side / float(w)))))
                else:
                    nh = max_side
                    nw = max(1, int(round(w * (max_side / float(h)))))

                resample = Image.Resampling.BILINEAR
                im = im.resize((nw, nh), resample=resample)

        a = np.asarray(im, dtype=np.uint8)

    x = a.astype(np.float32)
    x *= 1.0 / 255.0
    return x


def save_image(path: Path, img01: NDArray[np.float32]) -> None:
    x = np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)
    if x.ndim == 2:
        Image.fromarray(x, mode="L").save(path)
        return
    if x.ndim == 3 and x.shape[2] == 3:
        Image.fromarray(x, mode="RGB").save(path)
        return
    raise ValueError("unsupported image shape")
