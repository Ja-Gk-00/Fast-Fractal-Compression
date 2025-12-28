from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True)
class Preset:
    name: str
    params: dict[str, object]
    decode_iterations: int


TEST_DATA_DIR = Path(__file__).parent / "test-data"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMAGE_PATHS_ALL = [
    p
    for p in sorted(TEST_DATA_DIR.rglob("*"))
    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
]

if not IMAGE_PATHS_ALL:
    pytest.skip(f"no images found under {TEST_DATA_DIR}", allow_module_level=True)

_rng = np.random.default_rng(0)
_k = min(4, len(IMAGE_PATHS_ALL))
_idx = _rng.choice(len(IMAGE_PATHS_ALL), size=_k, replace=False)
IMAGE_PATHS = [IMAGE_PATHS_ALL[i] for i in sorted(_idx.tolist())]


PRESETS: tuple[Preset, ...] = (
    Preset(
        name="speed_quant_noqt",
        params={
            "max_block": 16,
            "min_block": 8,
            "stride": 4,
            "topk": 8,
            "entropy_thresh": 0.0,
            "max_domains": 128,
            "use_quadtree": False,
            "quantized": True,
        },
        decode_iterations=8,
    ),
    Preset(
        name="balanced_quant_qt",
        params={
            "max_block": 16,
            "min_block": 4,
            "stride": 2,
            "topk": 16,
            "entropy_thresh": 0.0,
            "max_domains": 256,
            "use_quadtree": True,
            "quantized": True,
        },
        decode_iterations=12,
    ),
    Preset(
        name="float_noqt",
        params={
            "max_block": 16,
            "min_block": 8,
            "stride": 4,
            "topk": 16,
            "entropy_thresh": 0.0,
            "max_domains": 256,
            "use_quadtree": False,
            "quantized": False,
        },
        decode_iterations=8,
    ),
)


def _filter_kwargs(fn: object, kwargs: dict[str, object]) -> dict[str, object]:
    sig = inspect.signature(fn)
    allowed = {
        k
        for k, p in sig.parameters.items()
        if p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


def _encode_kwargs(encode_array: object, p: Preset) -> dict[str, object]:
    sig = inspect.signature(encode_array)
    names = set(sig.parameters.keys())
    out: dict[str, object] = {}

    if "max_block" in names:
        out["max_block"] = p.params["max_block"]
    elif "block" in names:
        out["block"] = p.params["max_block"]

    if "min_block" in names:
        out["min_block"] = p.params["min_block"]

    if "stride" in names:
        out["stride"] = p.params["stride"]

    if "topk" in names:
        out["topk"] = p.params["topk"]

    if "entropy_thresh" in names:
        out["entropy_thresh"] = p.params["entropy_thresh"]

    if "max_domains" in names:
        out["max_domains"] = p.params["max_domains"]

    if "use_quadtree" in names:
        out["use_quadtree"] = p.params["use_quadtree"]

    if "quantized" in names:
        out["quantized"] = p.params["quantized"]

    return _filter_kwargs(encode_array, out)


def _load_img(p: Path, mode: Literal["L", "RGB"], max_side: int) -> NDArray[np.float32]:
    im = Image.open(p).convert(mode)
    w, h = im.size
    if max(w, h) > max_side:
        if w >= h:
            nw = max_side
            nh = max(1, int(round(h * (max_side / float(w)))))
        else:
            nh = max_side
            nw = max(1, int(round(w * (max_side / float(h)))))
        resample = (
            Image.Resampling.BILINEAR
            if hasattr(Image, "Resampling")
            else Image.BILINEAR
        )
        im = im.resize((nw, nh), resample=resample)

    a = np.asarray(im, dtype=np.uint8)
    x = (a.astype(np.float32) / 255.0).astype(np.float32, copy=False)
    return x if mode == "RGB" else x.astype(np.float32, copy=False)


def _make_non_multiple(x: NDArray[np.float32]) -> NDArray[np.float32]:
    if x.ndim == 2:
        h, w = int(x.shape[0]), int(x.shape[1])
        nh = max(17, h - 3) if h > 20 else max(17, h - 1)
        nw = max(17, w - 5) if w > 22 else max(17, w - 1)
        return x[:nh, :nw]
    h, w, c = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    nh = max(17, h - 3) if h > 20 else max(17, h - 1)
    nw = max(17, w - 5) if w > 22 else max(17, w - 1)
    return x[:nh, :nw, :c]


def _assert_reconstruction(src: NDArray[np.float32], rec: NDArray[np.float32]) -> None:
    assert rec.dtype == np.float32
    assert rec.shape == src.shape
    assert np.isfinite(rec).all()

    r = np.clip(rec, 0.0, 1.0)
    s = np.clip(src, 0.0, 1.0)
    mse = float(np.mean((r - s) ** 2))
    assert np.isfinite(mse)
    assert mse < 0.45


@pytest.mark.parametrize("img_path", IMAGE_PATHS, ids=lambda p: Path(p).name)
@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("shape_case", ["as_is", "non_multiple"])
@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_integration_encode_decode_array(
    img_path: Path,
    mode: Literal["L", "RGB"],
    shape_case: Literal["as_is", "non_multiple"],
    preset: Preset,
) -> None:
    from fastfractal.core.decode import decode_array
    from fastfractal.core.encode import encode_array

    x = _load_img(img_path, mode=mode, max_side=160)
    if shape_case == "non_multiple":
        x = _make_non_multiple(x)

    kwargs = _encode_kwargs(encode_array, preset)
    code = encode_array(x, **kwargs)
    rec = decode_array(code, iterations=int(preset.decode_iterations))

    if mode == "L" and rec.ndim == 3 and rec.shape[2] == 1:
        rec = rec[:, :, 0].astype(np.float32, copy=False)

    _assert_reconstruction(
        x.astype(np.float32, copy=False), rec.astype(np.float32, copy=False)
    )


@pytest.mark.parametrize("img_path", IMAGE_PATHS, ids=lambda p: Path(p).name)
@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_integration_encode_decode_file_pipeline(
    img_path: Path, preset: Preset, tmp_path: Path
) -> None:
    try:
        from fastfractal.core.decode import decode_to_file
        from fastfractal.core.encode import encode_to_file
    except Exception:
        pytest.skip(
            "encode_to_file/decode_to_file not available", allow_module_level=True
        )

    out_code = tmp_path / f"{img_path.stem}.{preset.name}.ffc"
    out_img = tmp_path / f"{img_path.stem}.{preset.name}.decoded.png"

    kwargs = _filter_kwargs(encode_to_file, _encode_kwargs(encode_to_file, preset))
    encode_to_file(img_path, out_code, **kwargs)
    decode_to_file(out_code, out_img, iterations=int(preset.decode_iterations))

    assert out_code.exists() and out_code.stat().st_size > 0
    assert out_img.exists() and out_img.stat().st_size > 0

    im = Image.open(out_img)
    assert im.size[0] > 0 and im.size[1] > 0
