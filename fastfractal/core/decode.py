from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastfractal._cext_backend import cext
from fastfractal.core.transforms import apply_transform_2d
from fastfractal.core.types import FractalCode
from fastfractal.io.codebook import load_code
from fastfractal.io.imageio import save_image


def downsample2x2(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if cext.has("downsample2x2"):
        return cext.call("downsample2x2", x)
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

    pool_blocks = np.ascontiguousarray(code.pool_blocks, dtype=np.uint16)
    pool_offsets = np.ascontiguousarray(code.pool_offsets, dtype=np.uint32)
    domain_yx = np.ascontiguousarray(code.domain_yx, dtype=np.uint16)

    leaf_yx = np.ascontiguousarray(code.leaf_yx, dtype=np.uint16)
    leaf_pool = np.ascontiguousarray(code.leaf_pool, dtype=np.uint8)
    leaf_dom = np.ascontiguousarray(code.leaf_dom, dtype=np.uint32)
    leaf_tf = np.ascontiguousarray(code.leaf_tf, dtype=np.uint8)

    quantized = bool(code.quantized)
    if quantized:
        if code.leaf_codes_q is None:
            raise ValueError("missing leaf_codes_q")
        codes_q = np.ascontiguousarray(code.leaf_codes_q, dtype=np.uint8)
        codes_f: NDArray[np.float32] | None = None
    else:
        if code.leaf_codes_f is None:
            raise ValueError("missing leaf_codes_f")
        codes_f = np.ascontiguousarray(code.leaf_codes_f, dtype=np.float32)
        codes_q = None

    if cext.has("decode_iter_f32"):
        if c == 1:
            cur0: NDArray[np.float32] = np.zeros((h, w), dtype=np.float32)
        else:
            cur0 = np.zeros((h, w, c), dtype=np.float32)

        out = cext.call(
            "decode_iter_f32",
            cur0,
            int(h),
            int(w),
            int(c),
            pool_blocks,
            pool_offsets,
            domain_yx,
            leaf_yx,
            leaf_pool,
            leaf_dom,
            leaf_tf,
            bool(quantized),
            codes_q if quantized else None,
            codes_f if not quantized else None,
            float(code.s_clip),
            float(code.o_min),
            float(code.o_max),
            int(iterations),
        )
        out = np.asarray(out, dtype=np.float32)

        if c == 1:
            return out[:oh, :ow].astype(np.float32, copy=False)
        return out[:oh, :ow, :].astype(np.float32, copy=False)

    if c == 1:
        cur: NDArray[np.float32] = np.zeros((h, w), dtype=np.float32)
    else:
        cur = np.zeros((h, w, c), dtype=np.float32)

    for _ in range(int(iterations)):
        nxt = np.zeros_like(cur)

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
                if quantized:
                    s = dequant_s(int(codes_q[i, 0, 0]), float(code.s_clip))
                    o = dequant_o(
                        int(codes_q[i, 0, 1]), float(code.o_min), float(code.o_max)
                    )
                else:
                    s = float(codes_f[i, 0, 0])
                    o = float(codes_f[i, 0, 1])
                blk = np.clip(np.float32(s) * dt + np.float32(o), 0.0, 1.0).astype(
                    np.float32, copy=False
                )
                nxt[y : y + b, x : x + b] = blk
            else:
                for ch in range(c):
                    dom = cur[dy : dy + 2 * b, dx : dx + 2 * b, ch]
                    ds = downsample2x2(dom)
                    dt = apply_transform_2d(ds, t)
                    if quantized:
                        s = dequant_s(int(codes_q[i, ch, 0]), float(code.s_clip))
                        o = dequant_o(
                            int(codes_q[i, ch, 1]), float(code.o_min), float(code.o_max)
                        )
                    else:
                        s = float(codes_f[i, ch, 0])
                        o = float(codes_f[i, ch, 1])
                    blk = np.clip(np.float32(s) * dt + np.float32(o), 0.0, 1.0).astype(
                        np.float32, copy=False
                    )
                    nxt[y : y + b, x : x + b, ch] = blk

        cur = nxt

    if c == 1:
        return cur[:oh, :ow].astype(np.float32, copy=False)
    return cur[:oh, :ow, :].astype(np.float32, copy=False)


def decode_to_file(input_path: Path, output_path: Path, iterations: int = 8) -> None:
    code = load_code(input_path)
    img = decode_array(code, iterations=int(iterations))
    save_image(output_path, img)


def decode(input_path: Path, output_path: Path, iterations: int = 8) -> None:
    decode_to_file(input_path, output_path, iterations=int(iterations))
