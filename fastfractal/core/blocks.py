import numpy as np

from fastfractal._cext_backend import cext


def domains_yx(h: int, w: int, block: int, stride: int) -> np.ndarray:
    if cext.has("domains_yx"):
        return cext.call("domains_yx", int(h), int(w), int(block), int(stride))
    return _domains_yx_py(h, w, block, stride)


def _domains_yx_py(h: int, w: int, block: int, stride: int) -> np.ndarray:
    if block <= 0:
        raise ValueError("block must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if block > h or block > w:
        return np.zeros((0, 2), dtype=np.uint16)

    lim_y = h - block
    lim_x = w - block
    ys = np.arange(0, lim_y + 1, stride, dtype=np.int32)
    xs = np.arange(0, lim_x + 1, stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    out = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)

    if out.size == 0:
        return np.zeros((0, 2), dtype=np.uint16)

    if out.max() > np.iinfo(np.uint16).max:
        raise ValueError("domains_yx coordinates exceed uint16 range")

    return out.astype(np.uint16, copy=False)


def extract_range(img: np.ndarray, y: int, x: int, block: int) -> np.ndarray:
    return img[y : y + block, x : x + block]


def ranges_yx(h: int, w: int, block: int) -> np.ndarray:
    if cext.has("ranges_yx"):
        return cext.call("ranges_yx", int(h), int(w), int(block))
    return _ranges_yx_py(h, w, block)


def _ranges_yx_py(h: int, w: int, block: int) -> np.ndarray:
    if block <= 0:
        raise ValueError("block must be > 0")
    if block > h or block > w:
        return np.zeros((0, 2), dtype=np.uint16)

    lim_y = h - block
    lim_x = w - block
    ys = np.arange(0, lim_y + 1, block, dtype=np.int32)
    xs = np.arange(0, lim_x + 1, block, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    out = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)

    if out.size == 0:
        return np.zeros((0, 2), dtype=np.uint16)

    if out.max() > np.iinfo(np.uint16).max:
        raise ValueError("ranges_yx coordinates exceed uint16 range")

    return out.astype(np.uint16, copy=False)
