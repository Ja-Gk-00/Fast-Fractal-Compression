from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Grid:
    height: int
    width: int
    block: int

    @property
    def ranges_h(self) -> int:
        return self.height // self.block

    @property
    def ranges_w(self) -> int:
        return self.width // self.block

    def iter_ranges(self) -> Iterator[tuple[int, int, int]]:
        idx = 0
        for ry in range(self.ranges_h):
            y = ry * self.block
            for rx in range(self.ranges_w):
                x = rx * self.block
                yield idx, y, x
                idx += 1


def iter_domains(
    height: int, width: int, block: int, stride: int
) -> Iterator[tuple[int, int, int]]:
    d = 2 * block
    idx = 0
    for y in range(0, height - d + 1, stride):
        for x in range(0, width - d + 1, stride):
            yield idx, y, x
            idx += 1


def extract_range(
    img: NDArray[np.float32], y: int, x: int, block: int
) -> NDArray[np.float32]:
    return img[y : y + block, x : x + block].astype(np.float32, copy=False)


def extract_domain(
    img: NDArray[np.float32], y: int, x: int, block: int
) -> NDArray[np.float32]:
    d = 2 * block
    return img[y : y + d, x : x + d].astype(np.float32, copy=False)
