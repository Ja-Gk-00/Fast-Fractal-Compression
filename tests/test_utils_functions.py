import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _import_any(names, err):
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(err)


def entropy_m():
    return _import_any(
        ("fastfractal.utils.entropy", "utils.entropy", "entropy"),
        "entropy module not found",
    )


def logger_m():
    return _import_any(
        ("fastfractal.utils.logger", "utils.logger", "logger"),
        "logger module not found",
    )


def image_process_m():
    return _import_any(
        ("fastfractal.utils.image_process", "utils.image_process", "image_process"),
        "image_process module not found",
    )


def visualize_m():
    return _import_any(
        ("fastfractal.utils.visualize", "utils.visualize", "visualize"),
        "visualize module not found",
    )


def test_entropy01_constant():
    m = entropy_m()
    x = np.zeros((16, 16), dtype=np.float32)
    assert m.entropy01(x) == 0.0


def test_entropy01_uniform_256():
    m = entropy_m()
    x = (np.arange(256, dtype=np.float32) / 255.0).reshape(16, 16)
    assert m.entropy01(x) == pytest.approx(1.0, abs=1e-6)


def test_entropy01_clips_out_of_range():
    m = entropy_m()
    x = np.array([-0.1, 1.2], dtype=np.float32)
    x = np.tile(x, 128).reshape(16, 16)
    assert m.entropy01(x) == pytest.approx(0.125, abs=1e-6)


def test_logger_emits():
    m = logger_m()
    out = []
    log = m.Logger(out.append)
    log.info("hello")
    assert out == ["hello"]


def _make_rgb_png(p: Path, size=(10, 10)):
    img = Image.new("RGB", size, (255, 0, 0))
    img.save(p, "PNG")
    return p


def test_process_image_rgb_resize_and_format(tmp_path: Path):
    m = image_process_m()
    inp = _make_rgb_png(tmp_path / "in.png", size=(10, 10))
    out = tmp_path / "out" / "nested" / "img.jpg"
    m.process_image(
        inp, out, size=(8, 6), grayscale=False, to_format="jpeg", jpg_quality=90
    )
    out2 = out.with_suffix(".jpeg")
    assert out2.exists()
    img2 = Image.open(out2)
    assert img2.size == (8, 6)
    assert img2.mode == "RGB"


def test_process_image_grayscale(tmp_path: Path):
    m = image_process_m()
    inp = _make_rgb_png(tmp_path / "in.png", size=(7, 5))
    out = tmp_path / "out.png"
    m.process_image(inp, out, grayscale=True, to_format="png")
    img2 = Image.open(out.with_suffix(".png"))
    assert img2.size == (7, 5)
    assert img2.mode == "L"


def test_process_image_uses_output_suffix_when_no_format(tmp_path: Path):
    m = image_process_m()
    inp = _make_rgb_png(tmp_path / "in.png", size=(9, 9))
    out = tmp_path / "out.jpeg"
    m.process_image(inp, out, to_format=None)
    assert out.exists()


def test_process_image_jpg_format_raises_in_pillow(tmp_path: Path):
    m = image_process_m()
    inp = _make_rgb_png(tmp_path / "in.png", size=(3, 3))
    out = tmp_path / "out.jpg"
    with pytest.raises((ValueError, KeyError, Image.UnidentifiedImageError)):
        m.process_image(inp, out, to_format="jpg")


def _install_fake_fastfractal(monkeypatch, decode_array, load_code):
    ff = types.ModuleType("fastfractal")
    core = types.ModuleType("fastfractal.core")
    dec = types.ModuleType("fastfractal.core.decode")
    dec.decode_array = decode_array
    io = types.ModuleType("fastfractal.io")
    cb = types.ModuleType("fastfractal.io.codebook")
    cb.load_code = load_code
    monkeypatch.setitem(sys.modules, "fastfractal", ff)
    monkeypatch.setitem(sys.modules, "fastfractal.core", core)
    monkeypatch.setitem(sys.modules, "fastfractal.core.decode", dec)
    monkeypatch.setitem(sys.modules, "fastfractal.io", io)
    monkeypatch.setitem(sys.modules, "fastfractal.io.codebook", cb)


def test_to_gray_u8_variants():
    m = visualize_m()
    a = np.zeros((2, 3), dtype=np.uint8)
    assert m._to_gray_u8(a) is a

    b = np.zeros((2, 3, 1), dtype=np.uint8)
    g = m._to_gray_u8(b)
    assert g.shape == (2, 3)
    assert np.array_equal(g, b[:, :, 0])

    c = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    g2 = m._to_gray_u8(c)
    assert g2.shape == (1, 3)
    assert int(g2[0, 0]) == 76
    assert int(g2[0, 1]) == 149
    assert int(g2[0, 2]) == 29

    with pytest.raises(ValueError):
        m._to_gray_u8(np.zeros((2, 2, 2), dtype=np.uint8))


def test_clamp_u8():
    m = visualize_m()
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    y = m._clamp_u8(x)
    assert np.array_equal(y, np.array([0, 0, 127, 255, 255], dtype=np.uint8))


def test_edge_mask_grid():
    m = visualize_m()
    h = 4
    w = 4
    leaf_yx = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.uint16)
    leaf_pool = np.array([0, 0, 0, 0], dtype=np.uint8)
    pool_blocks = np.array([2], dtype=np.uint16)

    mask = m._edge_mask(h, w, leaf_yx, leaf_pool, pool_blocks, 1)
    exp = np.zeros((h, w), dtype=bool)
    exp[0, :] = True
    exp[2, :] = True
    exp[:, 0] = True
    exp[:, 2] = True
    assert np.array_equal(mask, exp)

    mask2 = m._edge_mask(h, w, leaf_yx, leaf_pool, pool_blocks, 2)
    assert mask2.sum() > mask.sum()
    assert np.all(mask2[mask])


@dataclass
class _Code:
    height: int
    width: int
    orig_height: int
    orig_width: int
    leaf_yx: np.ndarray
    leaf_pool: np.ndarray
    pool_blocks: np.ndarray


def test_visualize_blocks_saves_and_overlays(tmp_path: Path, monkeypatch):
    m = visualize_m()

    def decode_array(code, iterations=8):
        return np.zeros((code.height, code.width, 3), dtype=np.float32)

    leaf_yx = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.uint16)
    leaf_pool = np.array([0, 0, 0, 0], dtype=np.uint8)
    pool_blocks = np.array([2], dtype=np.uint16)
    code = _Code(4, 4, 4, 4, leaf_yx, leaf_pool, pool_blocks)

    def load_code(x):
        return code

    _install_fake_fastfractal(
        monkeypatch, decode_array=decode_array, load_code=load_code
    )

    out = tmp_path / "blocks.png"
    p = m.visualize_blocks(
        code,
        out,
        background="decode",
        grayscale=True,
        thickness=1,
        line_value=200,
        alpha=1.0,
        upscale=1,
    )
    assert p == out
    arr = np.array(Image.open(out))
    assert arr.shape == (4, 4)
    assert int(arr[0, 0]) == 200
    assert int(arr[1, 1]) == 0


def test_visualize_blocks_from_file_uses_fallback_and_default_out(
    tmp_path: Path, monkeypatch
):
    m = visualize_m()
    code_path = tmp_path / "x.code"
    code_path.write_bytes(b"abc")

    class CodeObj:
        pass

    code_obj = CodeObj()

    def decode_array(code, iterations=8):
        return np.zeros((1, 1), dtype=np.float32)

    def load_code(arg):
        if isinstance(arg, Path):
            raise RuntimeError("force bytes path")
        return code_obj

    _install_fake_fastfractal(
        monkeypatch, decode_array=decode_array, load_code=load_code
    )

    called = {}

    def vb(code, out_path, **kw):
        called["code"] = code
        called["out"] = Path(out_path)
        return Path(out_path)

    monkeypatch.setattr(m, "visualize_blocks", vb)

    p = m.visualize_blocks_from_file(code_path)
    assert called["code"] is code_obj
    assert p == code_path.with_suffix(".blocks.png")
