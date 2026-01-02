from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from fastfractal.core.search import LSHIndex


@dataclass(frozen=True, slots=True)
class FractalCode:
    height: int
    width: int
    orig_height: int
    orig_width: int
    channels: int

    pool_blocks: NDArray[np.uint16]
    pool_strides: NDArray[np.uint16]
    pool_offsets: NDArray[np.uint32]
    domain_yx: NDArray[np.uint16]

    leaf_yx: NDArray[np.uint16]
    leaf_pool: NDArray[np.uint8]
    leaf_dom: NDArray[np.uint32]
    leaf_tf: NDArray[np.uint8]

    quantized: bool
    s_clip: float
    o_min: float
    o_max: float
    leaf_codes_q: NDArray[np.uint8] | None
    leaf_codes_f: NDArray[np.float32] | None

    iterations_hint: int


@dataclass(frozen=True, slots=True)
class PoolRuntime:
    block: int
    stride: int
    domain_yx: NDArray[np.uint16]
    tf_flat: NDArray[np.float32]

    tf_sum: NDArray[np.float64] | None
    tf_sum2: NDArray[np.float64] | None

    proxy_mat: NDArray[np.float32]
    map_dom: NDArray[np.uint32]
    map_tf: NDArray[np.uint8]
    entry_bucket: NDArray[np.uint8]
    bucket_entries: list[NDArray[np.int64]]
    backend: str
    lsh: LSHIndex | None
    pca_mean: NDArray[np.float32] | None
    pca_basis: NDArray[np.float32] | None
    transform_ids: tuple[int, ...] | None


MAGIC_V2: Final[bytes] = b"FFC2"
