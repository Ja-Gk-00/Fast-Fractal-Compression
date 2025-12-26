from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastfractal.core.types import FractalCode, MAGIC_V2


_HDR = "<4sIIIIBHIIB"
_HDR_SZ = struct.calcsize(_HDR)
_QHDR = "<fff"
_QHDR_SZ = struct.calcsize(_QHDR)


def save_code(path: Path, code: FractalCode) -> None:
    h = int(code.height)
    w = int(code.width)
    oh = int(code.orig_height)
    ow = int(code.orig_width)
    c = int(code.channels)
    pools = int(code.pool_blocks.shape[0])
    leaves = int(code.leaf_yx.shape[0])
    it = int(code.iterations_hint)
    flags = 1 if code.quantized else 0

    header = struct.pack(_HDR, MAGIC_V2, h, w, oh, ow, c, pools, leaves, it, flags)
    qhdr = struct.pack(_QHDR, float(code.s_clip), float(code.o_min), float(code.o_max))

    pool_blocks = code.pool_blocks.astype(np.uint16, copy=False)
    pool_strides = code.pool_strides.astype(np.uint16, copy=False)
    pool_offsets = code.pool_offsets.astype(np.uint32, copy=False)

    domain_yx = code.domain_yx.astype(np.uint16, copy=False)

    leaf_yx = code.leaf_yx.astype(np.uint16, copy=False)
    leaf_pool = code.leaf_pool.astype(np.uint8, copy=False)
    leaf_dom = code.leaf_dom.astype(np.uint32, copy=False)
    leaf_tf = code.leaf_tf.astype(np.uint8, copy=False)

    with path.open("wb") as f:
        f.write(header)
        f.write(qhdr)

        f.write(pool_blocks.tobytes(order="C"))
        f.write(pool_strides.tobytes(order="C"))
        f.write(pool_offsets.tobytes(order="C"))

        f.write(domain_yx.tobytes(order="C"))

        f.write(leaf_yx.tobytes(order="C"))
        f.write(leaf_pool.tobytes(order="C"))
        f.write(leaf_dom.tobytes(order="C"))
        f.write(leaf_tf.tobytes(order="C"))

        if code.quantized:
            if code.leaf_codes_q is None:
                raise ValueError("leaf_codes_q missing")
            f.write(code.leaf_codes_q.astype(np.uint8, copy=False).tobytes(order="C"))
        else:
            if code.leaf_codes_f is None:
                raise ValueError("leaf_codes_f missing")
            f.write(code.leaf_codes_f.astype(np.float32, copy=False).tobytes(order="C"))


def load_code(path: Path) -> FractalCode:
    with path.open("rb") as f:
        header = f.read(_HDR_SZ)
        magic, h, w, oh, ow, c, pools, leaves, it, flags = struct.unpack(_HDR, header)
        if magic != MAGIC_V2:
            raise ValueError("bad magic")

        s_clip, o_min, o_max = struct.unpack(_QHDR, f.read(_QHDR_SZ))
        quantized = (int(flags) & 1) != 0

        pool_blocks = np.frombuffer(f.read(int(pools) * 2), dtype=np.uint16).copy()
        pool_strides = np.frombuffer(f.read(int(pools) * 2), dtype=np.uint16).copy()
        pool_offsets = np.frombuffer(f.read((int(pools) + 1) * 4), dtype=np.uint32).copy()

        total_domains = int(pool_offsets[-1])
        domain_yx = (
            np.frombuffer(f.read(total_domains * 4), dtype=np.uint16)
            .reshape(total_domains, 2)
            .copy()
        )

        leaf_yx = (
            np.frombuffer(f.read(int(leaves) * 4), dtype=np.uint16)
            .reshape(int(leaves), 2)
            .copy()
        )
        leaf_pool = np.frombuffer(f.read(int(leaves)), dtype=np.uint8).copy()
        leaf_dom = np.frombuffer(f.read(int(leaves) * 4), dtype=np.uint32).copy()
        leaf_tf = np.frombuffer(f.read(int(leaves)), dtype=np.uint8).copy()

        leaf_codes_q: NDArray[np.uint8] | None = None
        leaf_codes_f: NDArray[np.float32] | None = None
        if quantized:
            n = int(leaves) * int(c) * 2
            leaf_codes_q = (
                np.frombuffer(f.read(n), dtype=np.uint8)
                .reshape(int(leaves), int(c), 2)
                .copy()
            )
        else:
            n = int(leaves) * int(c) * 2
            leaf_codes_f = (
                np.frombuffer(f.read(n * 4), dtype=np.float32)
                .reshape(int(leaves), int(c), 2)
                .copy()
            )

    return FractalCode(
        height=int(h),
        width=int(w),
        orig_height=int(oh),
        orig_width=int(ow),
        channels=int(c),
        pool_blocks=pool_blocks,
        pool_strides=pool_strides,
        pool_offsets=pool_offsets,
        domain_yx=domain_yx,
        leaf_yx=leaf_yx,
        leaf_pool=leaf_pool,
        leaf_dom=leaf_dom,
        leaf_tf=leaf_tf,
        quantized=bool(quantized),
        s_clip=float(s_clip),
        o_min=float(o_min),
        o_max=float(o_max),
        leaf_codes_q=leaf_codes_q,
        leaf_codes_f=leaf_codes_f,
        iterations_hint=int(it),
    )
