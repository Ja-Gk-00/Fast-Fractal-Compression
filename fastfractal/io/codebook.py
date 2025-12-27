from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastfractal.core.types import MAGIC_V2, FractalCode

_HDR = "<4sIIIIBHIIB"
_HDR_SZ = struct.calcsize(_HDR)
_QHDR = "<fff"
_QHDR_SZ = struct.calcsize(_QHDR)


def dump_code(code: FractalCode) -> bytes:
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

    chunks: list[bytes] = [
        header,
        qhdr,
        pool_blocks.tobytes(order="C"),
        pool_strides.tobytes(order="C"),
        pool_offsets.tobytes(order="C"),
        domain_yx.tobytes(order="C"),
        leaf_yx.tobytes(order="C"),
        leaf_pool.tobytes(order="C"),
        leaf_dom.tobytes(order="C"),
        leaf_tf.tobytes(order="C"),
    ]

    if code.quantized:
        if code.leaf_codes_q is None:
            raise ValueError("leaf_codes_q missing")
        chunks.append(code.leaf_codes_q.astype(np.uint8, copy=False).tobytes(order="C"))
    else:
        if code.leaf_codes_f is None:
            raise ValueError("leaf_codes_f missing")
        chunks.append(
            code.leaf_codes_f.astype(np.float32, copy=False).tobytes(order="C")
        )

    return b"".join(chunks)


def load_code_bytes(data: bytes) -> FractalCode:
    mv = memoryview(data)
    if len(mv) < _HDR_SZ + _QHDR_SZ:
        raise ValueError("truncated")

    magic, h, w, oh, ow, c, pools, leaves, it, flags = struct.unpack_from(_HDR, mv, 0)
    if magic != MAGIC_V2:
        raise ValueError("bad magic")

    s_clip, o_min, o_max = struct.unpack_from(_QHDR, mv, _HDR_SZ)
    quantized = (int(flags) & 1) != 0

    off = _HDR_SZ + _QHDR_SZ

    def take(n: int) -> memoryview:
        nonlocal off
        if off + n > len(mv):
            raise ValueError("truncated")
        out = mv[off : off + n]
        off += n
        return out

    pool_blocks = np.frombuffer(take(int(pools) * 2), dtype=np.uint16).copy()
    pool_strides = np.frombuffer(take(int(pools) * 2), dtype=np.uint16).copy()
    pool_offsets = np.frombuffer(take((int(pools) + 1) * 4), dtype=np.uint32).copy()

    total_domains = int(pool_offsets[-1])
    domain_yx = (
        np.frombuffer(take(total_domains * 4), dtype=np.uint16)
        .reshape(total_domains, 2)
        .copy()
    )

    leaf_yx = (
        np.frombuffer(take(int(leaves) * 4), dtype=np.uint16)
        .reshape(int(leaves), 2)
        .copy()
    )
    leaf_pool = np.frombuffer(take(int(leaves)), dtype=np.uint8).copy()
    leaf_dom = np.frombuffer(take(int(leaves) * 4), dtype=np.uint32).copy()
    leaf_tf = np.frombuffer(take(int(leaves)), dtype=np.uint8).copy()

    leaf_codes_q: NDArray[np.uint8] | None = None
    leaf_codes_f: NDArray[np.float32] | None = None
    if quantized:
        n = int(leaves) * int(c) * 2
        leaf_codes_q = (
            np.frombuffer(take(n), dtype=np.uint8)
            .reshape(int(leaves), int(c), 2)
            .copy()
        )
    else:
        n = int(leaves) * int(c) * 2
        leaf_codes_f = (
            np.frombuffer(take(n * 4), dtype=np.float32)
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


def save_code(path: Path, code: FractalCode) -> None:
    path.write_bytes(dump_code(code))


def load_code(path: Path) -> FractalCode:
    return load_code_bytes(path.read_bytes())
