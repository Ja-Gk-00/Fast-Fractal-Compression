import numpy as np
from numpy.typing import NDArray


def _clipf(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def dequant_s(q: int, s_clip: float) -> float:
    return float(q) * (2.0 * s_clip) / 255.0 - s_clip


def dequant_o(q: int, o_min: float, o_max: float) -> float:
    return o_min + float(q) * (o_max - o_min) / 255.0


def quant_s(s: float, s_clip: float) -> int:
    sc = _clipf(s, -float(s_clip), float(s_clip))
    q = int(round((sc + float(s_clip)) * 255.0 / (2.0 * float(s_clip))))
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def quant_o(o: float, o_min: float, o_max: float) -> int:
    oc = _clipf(o, float(o_min), float(o_max))
    q = int(round((oc - float(o_min)) * 255.0 / (float(o_max) - float(o_min))))
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def rgb_to_luma(img: NDArray[np.float32]) -> NDArray[np.float32]:
    if img.ndim == 2:
        return img
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return np.asarray(0.2989 * r + 0.5870 * g + 0.1140 * b, dtype=np.float32)
