from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastfractal import _cext  # type: ignore
from fastfractal.core.transforms import apply_transform_2d
from fastfractal.core.types import FractalCode
from fastfractal.io.codebook import load_code
from fastfractal.io.imageio import save_image

_HAS_DOWNSAMPLE = hasattr(_cext, "downsample2x2")


def downsample2x2(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if _HAS_DOWNSAMPLE:
        return _cext.downsample2x2(x)  # type: ignore[no-any-return]
    return (
        (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])
        * np.float32(0.25)
    ).astype(np.float32, copy=False)


def dequant_s(q: int, s_clip: float) -> float:
    return float(q) * (2.0 * s_clip) / 255.0 - s_clip


def dequant_o(q: int, o_min: float, o_max: float) -> float:
    return o_min + float(q) * (o_max - o_min) / 255.0


def _downsample_image2x2(cur: NDArray[np.float32]) -> NDArray[np.float32]:
    if cur.ndim == 2:
        if _HAS_DOWNSAMPLE:
            return _cext.downsample2x2(cur)  # type: ignore[no-any-return]
        return (
            (cur[0::2, 0::2] + cur[1::2, 0::2] + cur[0::2, 1::2] + cur[1::2, 1::2])
            * np.float32(0.25)
        ).astype(np.float32, copy=False)

    out = cur[0::2, 0::2, :].astype(np.float32, copy=True)
    out += cur[1::2, 0::2, :]
    out += cur[0::2, 1::2, :]
    out += cur[1::2, 1::2, :]
    out *= np.float32(0.25)
    return out


def decode_array(code: FractalCode, iterations: int = 8) -> NDArray[np.float32]:
    h = int(code.height)
    w = int(code.width)
    oh = int(code.orig_height)
    ow = int(code.orig_width)
    c = int(code.channels)

    pool_blocks = code.pool_blocks
    pool_offsets = code.pool_offsets
    domain_yx = code.domain_yx

    leaf_yx = code.leaf_yx
    leaf_pool = code.leaf_pool
    leaf_dom = code.leaf_dom
    leaf_tf = code.leaf_tf

    if code.quantized:
        if code.leaf_codes_q is None:
            raise ValueError("missing leaf_codes_q")
        codes_q = code.leaf_codes_q
    else:
        if code.leaf_codes_f is None:
            raise ValueError("missing leaf_codes_f")
        codes_f = code.leaf_codes_f

    nleaf = int(leaf_yx.shape[0])

    leaf_y = leaf_yx[:, 0].astype(np.intp, copy=False)
    leaf_x = leaf_yx[:, 1].astype(np.intp, copy=False)
    leaf_pi = leaf_pool.astype(np.intp, copy=False)
    leaf_t = leaf_tf.astype(np.intp, copy=False)
    leaf_b = pool_blocks[leaf_pi].astype(np.intp, copy=False)

    gd = pool_offsets[leaf_pi].astype(np.int64, copy=False) + leaf_dom.astype(
        np.int64, copy=False
    )
    dom_xy = domain_yx[gd.astype(np.intp, copy=False)]
    dom_y = dom_xy[:, 0].astype(np.intp, copy=False)
    dom_x = dom_xy[:, 1].astype(np.intp, copy=False)

    dom_y2 = (dom_y >> 1).astype(np.intp, copy=False)
    dom_x2 = (dom_x >> 1).astype(np.intp, copy=False)

    odd_mask = ((dom_y | dom_x) & 1) != 0

    if code.quantized:
        s_clip = np.float32(code.s_clip)
        o_min = np.float32(code.o_min)
        o_max = np.float32(code.o_max)
        s_all = (
            codes_q[:, :, 0].astype(np.float32)
            * (np.float32(2.0) * s_clip)
            / np.float32(255.0)
        ) - s_clip
        o_all = o_min + codes_q[:, :, 1].astype(np.float32) * (
            o_max - o_min
        ) / np.float32(255.0)
    else:
        s_all = codes_f[:, :, 0].astype(np.float32, copy=False)
        o_all = codes_f[:, :, 1].astype(np.float32, copy=False)

    if c == 1:
        cur: NDArray[np.float32] = np.zeros((h, w), dtype=np.float32)
        nxt: NDArray[np.float32] = np.empty((h, w), dtype=np.float32)
    else:
        cur = np.zeros((h, w, c), dtype=np.float32)
        nxt = np.empty((h, w, c), dtype=np.float32)

    for _ in range(int(iterations)):
        cur_ds = _downsample_image2x2(cur)

        nxt.fill(np.float32(0.0))

        if c == 1:
            s0 = s_all[:, 0]
            o0 = o_all[:, 0]

            for i in range(nleaf):
                y = leaf_y[i]
                x = leaf_x[i]
                b = leaf_b[i]
                t = int(leaf_t[i])

                if odd_mask[i]:
                    dy = dom_y[i]
                    dx = dom_x[i]
                    ds = downsample2x2(cur[dy : dy + 2 * b, dx : dx + 2 * b])
                else:
                    ds = cur_ds[dom_y2[i] : dom_y2[i] + b, dom_x2[i] : dom_x2[i] + b]

                dt = apply_transform_2d(ds, t)
                out = nxt[y : y + b, x : x + b]
                np.multiply(dt, s0[i], out=out)
                out += o0[i]
                np.clip(out, np.float32(0.0), np.float32(1.0), out=out)

        else:
            for i in range(nleaf):
                y = leaf_y[i]
                x = leaf_x[i]
                b = leaf_b[i]
                t = int(leaf_t[i])

                if odd_mask[i]:
                    dy = dom_y[i]
                    dx = dom_x[i]
                    for ch in range(c):
                        ds = downsample2x2(cur[dy : dy + 2 * b, dx : dx + 2 * b, ch])
                        dt = apply_transform_2d(ds, t)
                        out = nxt[y : y + b, x : x + b, ch]
                        np.multiply(dt, s_all[i, ch], out=out)
                        out += o_all[i, ch]
                        np.clip(out, np.float32(0.0), np.float32(1.0), out=out)
                    continue
                ds3 = cur_ds[dom_y2[i] : dom_y2[i] + b, dom_x2[i] : dom_x2[i] + b, :]

                for ch in range(c):
                    dt = apply_transform_2d(ds3[:, :, ch], t)
                    out = nxt[y : y + b, x : x + b, ch]
                    np.multiply(dt, s_all[i, ch], out=out)
                    out += o_all[i, ch]
                    np.clip(out, np.float32(0.0), np.float32(1.0), out=out)

        cur, nxt = nxt, cur

    if c == 1:
        return cur[:oh, :ow].astype(np.float32, copy=False)
    return cur[:oh, :ow, :].astype(np.float32, copy=False)


def decode_to_file(input_path: Path, output_path: Path, iterations: int = 8) -> None:
    code = load_code(input_path)
    img = decode_array(code, iterations=iterations)
    save_image(output_path, img)
