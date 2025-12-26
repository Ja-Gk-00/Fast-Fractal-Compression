from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastfractal import _cext  # type: ignore
from fastfractal.core.transforms import apply_transform_2d
from fastfractal.core.types import FractalCode
from fastfractal.io.codebook import load_code
from fastfractal.io.imageio import save_image


def _has_cext() -> bool:
    return hasattr(_cext, "downsample2x2")


def downsample2x2(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if _has_cext():
        return _cext.downsample2x2(x)  # type: ignore[no-any-return]
    h, w = x.shape
    return (
        x.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.float32, copy=False)
    )


def dequant_s(q: int, s_clip: float) -> float:
    return float(q) * (2.0 * s_clip) / 255.0 - s_clip


def dequant_o(q: int, o_min: float, o_max: float) -> float:
    return o_min + float(q) * (o_max - o_min) / 255.0


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
        codes_f: NDArray[np.float32] | None = None
    else:
        if code.leaf_codes_f is None:
            raise ValueError("missing leaf_codes_f")
        codes_f = code.leaf_codes_f
        codes_q = None

    if c == 1:
        cur: NDArray[np.float32] = np.zeros((h, w), dtype=np.float32)
    else:
        cur = np.zeros((h, w, c), dtype=np.float32)

    for _ in range(iterations):
        if c == 1:
            nxt: NDArray[np.float32] = np.zeros((h, w), dtype=np.float32)
        else:
            nxt = np.zeros((h, w, c), dtype=np.float32)

        for i in range(int(leaf_yx.shape[0])):
            y = int(leaf_yx[i, 0])
            x = int(leaf_yx[i, 1])
            pi = int(leaf_pool[i])
            b = int(pool_blocks[pi])
            di = int(leaf_dom[i])
            t = int(leaf_tf[i])

            gd = int(pool_offsets[pi]) + di
            dy = int(domain_yx[gd, 0])
            dx = int(domain_yx[gd, 1])

            if c == 1:
                dom = cur[dy : dy + 2 * b, dx : dx + 2 * b]
                ds = downsample2x2(dom)
                dt = apply_transform_2d(ds, t)
                if code.quantized:
                    s = dequant_s(int(codes_q[i, 0, 0]), float(code.s_clip))
                    o = dequant_o(int(codes_q[i, 0, 1]), float(code.o_min), float(code.o_max))
                else:
                    s = float(codes_f[i, 0, 0])
                    o = float(codes_f[i, 0, 1])
                blk = np.clip(np.float32(s) * dt + np.float32(o), 0.0, 1.0).astype(np.float32, copy=False)
                nxt[y : y + b, x : x + b] = blk
            else:
                for ch in range(c):
                    dom = cur[dy : dy + 2 * b, dx : dx + 2 * b, ch]
                    ds = downsample2x2(dom)
                    dt = apply_transform_2d(ds, t)
                    if code.quantized:
                        s = dequant_s(int(codes_q[i, ch, 0]), float(code.s_clip))
                        o = dequant_o(int(codes_q[i, ch, 1]), float(code.o_min), float(code.o_max))
                    else:
                        s = float(codes_f[i, ch, 0])
                        o = float(codes_f[i, ch, 1])
                    blk = np.clip(np.float32(s) * dt + np.float32(o), 0.0, 1.0).astype(np.float32, copy=False)
                    nxt[y : y + b, x : x + b, ch] = blk

        cur = nxt

    if c == 1:
        return cur[:oh, :ow].astype(np.float32, copy=False)
    return cur[:oh, :ow, :].astype(np.float32, copy=False)


def decode_to_file(input_path: Path, output_path: Path, iterations: int = 8) -> None:
    code = load_code(input_path)
    img = decode_array(code, iterations=iterations)
    save_image(output_path, img)
