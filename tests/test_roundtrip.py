from __future__ import annotations

import numpy as np

from fastfractal.core.decode import decode_array
from fastfractal.core.encode import encode_array


def test_roundtrip_gray_shape() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((32, 32), dtype=np.float32)
    code = encode_array(
        img, block=4, stride=2, topk=16, entropy_thresh=0.0, max_domains=128
    )
    out = decode_array(code, iterations=6)
    assert out.shape == img.shape


def test_roundtrip_rgb_shape() -> None:
    rng = np.random.default_rng(1)
    img = rng.random((32, 32, 3), dtype=np.float32)
    code = encode_array(
        img, block=4, stride=2, topk=16, entropy_thresh=0.0, max_domains=128
    )
    out = decode_array(code, iterations=6)
    assert out.shape == img.shape


def test_roundtrip_gray_shape_non_multiple() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((35, 51), dtype=np.float32)
    code = encode_array(
        img, block=8, stride=4, topk=16, entropy_thresh=0.0, max_domains=128
    )
    out = decode_array(code, iterations=6)
    assert out.shape == img.shape
