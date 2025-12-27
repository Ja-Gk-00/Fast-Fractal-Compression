from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _to_gray_u8(x: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[2] == 1:
        return x[:, :, 0]
    if x.ndim == 3 and x.shape[2] == 3:
        y = (
            0.299 * x[:, :, 0].astype(np.float32)
            + 0.587 * x[:, :, 1].astype(np.float32)
            + 0.114 * x[:, :, 2].astype(np.float32)
        )
        return np.clip(y, 0.0, 255.0).astype(np.uint8)
    raise ValueError("unsupported image shape for grayscale conversion")


def _require_pillow() -> object:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required for visualization: uv add pillow") from e
    return Image


def _clamp_u8(x: NDArray[np.float32]) -> NDArray[np.uint8]:
    y = np.clip(x, 0.0, 1.0) * 255.0
    return y.astype(np.uint8, copy=False)


def _edge_mask(
    h: int,
    w: int,
    leaf_yx: NDArray[np.uint16],
    leaf_pool: NDArray[np.uint8],
    pool_blocks: NDArray[np.uint16],
    thickness: int,
) -> NDArray[np.bool_]:
    y = leaf_yx[:, 0].astype(np.int32, copy=False)
    x = leaf_yx[:, 1].astype(np.int32, copy=False)
    b = pool_blocks[leaf_pool.astype(np.int32, copy=False)].astype(np.int32, copy=False)

    y2 = y + b
    x2 = x + b

    hdiff = np.zeros((h + 1, w + 1), dtype=np.int32)
    vdiff = np.zeros((h + 1, w + 1), dtype=np.int32)

    rr = np.concatenate([y, y2], axis=0)
    xs = np.concatenate([x, x], axis=0)
    xe = np.concatenate([x2, x2], axis=0)
    m = (rr >= 0) & (rr <= h) & (xs >= 0) & (xe >= 0) & (xs <= w) & (xe <= w)
    rr2 = rr[m]
    xs2 = xs[m]
    xe2 = xe[m]
    np.add.at(hdiff, (rr2, xs2), 1)
    np.add.at(hdiff, (rr2, xe2), -1)

    cc = np.concatenate([x, x2], axis=0)
    ys = np.concatenate([y, y], axis=0)
    ye = np.concatenate([y2, y2], axis=0)
    m2 = (cc >= 0) & (cc <= w) & (ys >= 0) & (ye >= 0) & (ys <= h) & (ye <= h)
    cc2 = cc[m2]
    ys2 = ys[m2]
    ye2 = ye[m2]
    np.add.at(vdiff, (ys2, cc2), 1)
    np.add.at(vdiff, (ye2, cc2), -1)

    hmask = np.cumsum(hdiff[:h, :w], axis=1) != 0
    vmask = np.cumsum(vdiff[:h, :w], axis=0) != 0
    mask = hmask | vmask

    t = int(thickness)
    if t <= 1:
        return mask

    out = mask.copy()
    for k in range(1, t):
        out[k:, :] |= mask[:-k, :]
        out[:-k, :] |= mask[k:, :]
        out[:, k:] |= mask[:, :-k]
        out[:, :-k] |= mask[:, k:]

    return out


def visualize_blocks(
    code: object,
    out_path: str | Path | None = None,
    *,
    background: Literal["decode", "black"] = "decode",
    grayscale: bool = True,
    iterations: int = 8,
    thickness: int = 1,
    line_value: int = 255,
    alpha: float = 1.0,
    upscale: int = 1,
) -> Path:
    from fastfractal.core.decode import decode_array

    code2 = code
    h = int(code2.height)
    w = int(code2.width)
    oh = int(code2.orig_height)
    ow = int(code2.orig_width)

    leaf_yx = code2.leaf_yx
    leaf_pool = code2.leaf_pool
    pool_blocks = code2.pool_blocks

    mask = _edge_mask(h, w, leaf_yx, leaf_pool, pool_blocks, int(thickness))

    u8: NDArray[np.uint8]
    if background == "decode":
        imgf = decode_array(code2, iterations=int(iterations))
        u8 = _clamp_u8(imgf.astype(np.float32, copy=False))
        if grayscale:
            u8 = _to_gray_u8(u8)
    elif background == "black":
        u8 = (
            np.zeros((h, w), dtype=np.uint8)
            if grayscale
            else np.zeros((h, w, 3), dtype=np.uint8)
        )
    else:
        raise ValueError("background must be 'decode' or 'black'")

    if oh < h or ow < w:
        mask = mask[:oh, :ow]
        if u8.ndim == 2:
            u8 = u8[:oh, :ow]
        else:
            u8 = u8[:oh, :ow, :]
        h = oh
        w = ow

    lv = int(np.clip(int(line_value), 0, 255))
    a = float(alpha)
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0

    if u8.ndim == 2:
        if a >= 1.0:
            u8[mask] = np.uint8(lv)
        else:
            base = u8.astype(np.float32, copy=False)
            base[mask] = base[mask] * (1.0 - a) + float(lv) * a
            u8 = np.clip(base, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        if a >= 1.0:
            u8[mask, :] = np.uint8(lv)
        else:
            base3 = u8.astype(np.float32, copy=False)
            base3[mask, :] = base3[mask, :] * (1.0 - a) + float(lv) * a
            u8 = np.clip(base3, 0.0, 255.0).astype(np.uint8, copy=False)

    p = Path(out_path) if out_path is not None else Path("blocks.png")
    Image = _require_pillow()

    if u8.ndim == 2:
        im = Image.fromarray(u8, mode="L")
    else:
        im = Image.fromarray(u8, mode="RGB")

    s = int(upscale)
    if s > 1:
        resample = (
            Image.Resampling.NEAREST
            if hasattr(Image, "Resampling")
            else Image.NEAREST  # ignore [attr-defined]
        )
        im = im.resize((w * s, h * s), resample=resample)

    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p)
    return p


def visualize_blocks_from_file(
    code_path: str | Path,
    out_path: str | Path | None = None,
    *,
    background: Literal["decode", "black"] = "decode",
    grayscale: bool = True,
    iterations: int = 8,
    thickness: int = 1,
    line_value: int = 255,
    alpha: float = 1.0,
    upscale: int = 1,
) -> Path:
    cp = Path(code_path)

    try:
        from fastfractal.io.codebook import load_code as _load_code_path

        code = _load_code_path(cp)
    except Exception:
        from fastfractal.io.codebook import load_code as _load_code_bytes

        code = _load_code_bytes(cp.read_bytes())

    out = cp.with_suffix(".blocks.png") if out_path is None else Path(out_path)

    return visualize_blocks(
        code,
        out,
        background=background,
        grayscale=grayscale,
        iterations=iterations,
        thickness=thickness,
        line_value=line_value,
        alpha=alpha,
        upscale=upscale,
    )
