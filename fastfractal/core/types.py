from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray


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

    @property
    def code(self):
        return self.leaf_codes_f if self.leaf_codes_f is not None else self.leaf_codes_q


MAGIC_V2: Final[bytes] = b"FFC2"
