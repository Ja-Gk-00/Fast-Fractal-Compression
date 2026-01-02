from __future__ import annotations

import inspect
import os
import random
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


def _select_images(paths: list[Path], k: int, seed: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) <= k:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, k)


IMAGE_PATHS_ALL = [
    p
    for p in sorted(TEST_DATA_DIR.rglob("*"))
    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
]

MAX_IMAGES = int(os.getenv("FASTFRACTAL_TEST_IMAGES", "4"))
SEED = int(os.getenv("FASTFRACTAL_TEST_SEED", "0"))
IMAGE_PATHS = _select_images(IMAGE_PATHS_ALL, k=max(1, min(MAX_IMAGES, 4)), seed=SEED)

if not IMAGE_PATHS:
    pytest.skip(f"no images found under {TEST_DATA_DIR}", allow_module_level=True)


def _require_encode_array():
    try:
        from fastfractal.core import encode as encode_mod  # noqa: WPS433
        from fastfractal.core.encode import encode_array  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        pytest.skip(f"fastfractal encoder not importable: {e}")
    return encode_array, encode_mod


def _available_backends(encode_mod: object) -> set[str]:
    sb = getattr(encode_mod, "SearchBackend", None)
    if sb is None:
        return set()

    if isinstance(sb, (set, frozenset, list, tuple)):
        return {str(x).lower() for x in sb}

    if hasattr(sb, "__members__"):
        out: set[str] = set()
        try:
            for m in sb:  # type: ignore[assignment]
                out.add(str(getattr(m, "value", m)).lower())
                out.add(str(getattr(m, "name", m)).lower())
        except TypeError:
            pass
        out.update({str(k).lower() for k in getattr(sb, "__members__", {}).keys()})
        return {x for x in out if x}

    try:
        return {str(x).lower() for x in sb}  # type: ignore[operator]
    except TypeError:
        return set()


def _filter_kwargs(fn: object, kwargs: dict[str, object]) -> dict[str, object]:
    sig = inspect.signature(fn)
    allowed = {
        k
        for k, p in sig.parameters.items()
        if p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


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


def _assert_reconstruction(src: NDArray[np.float32], rec: NDArray[np.float32]) -> None:
    assert rec.dtype == np.float32
    assert rec.shape == src.shape
    assert np.isfinite(rec).all()

    r = np.clip(rec, 0.0, 1.0)
    s = np.clip(src, 0.0, 1.0)
    mse = float(np.mean((r - s) ** 2))
    assert np.isfinite(mse)
    assert mse < 0.55


_encode_array, _encode_mod = _require_encode_array()
_SIG = inspect.signature(_encode_array)
_PARAM_NAMES = set(_SIG.parameters.keys())
_ALLOWED_BACKENDS = _available_backends(_encode_mod)


def _mk_preset(
    name: str, extra: dict[str, object], decode_iterations: int = 10
) -> Preset:
    base = {
        "max_block": 16,
        "min_block": 8,
        "stride": 4,
        "topk": 16,
        "entropy_thresh": 0.0,
        "max_domains": 256,
        "use_quadtree": False,
        "quantized": True,
        "iterations_hint": 8,
    }
    base.update(extra)
    return Preset(
        name=name,
        params=_filter_kwargs(_encode_array, base),
        decode_iterations=decode_iterations,
    )


PRESETS_LIST: list[Preset] = []

if "pca_dim" in _PARAM_NAMES:
    PRESETS_LIST.append(_mk_preset("adv_pca_dim16", {"pca_dim": 16}))

if "use_buckets" in _PARAM_NAMES:
    extra = {"use_buckets": True}
    if "bucket_count" in _PARAM_NAMES:
        extra["bucket_count"] = 8
    PRESETS_LIST.append(_mk_preset("adv_buckets", extra))

if "use_s_sets" in _PARAM_NAMES:
    PRESETS_LIST.append(_mk_preset("adv_s_sets", {"use_s_sets": True}))

if "backend" in _PARAM_NAMES and "lsh" in _ALLOWED_BACKENDS:
    extra = {"backend": "lsh"}
    if "lsh_planes" in _PARAM_NAMES:
        extra["lsh_planes"] = 16
    if "lsh_budget" in _PARAM_NAMES:
        extra["lsh_budget"] = 2048
    PRESETS_LIST.append(_mk_preset("adv_backend_lsh", extra))

PRESETS: tuple[Preset, ...] = tuple(PRESETS_LIST)

if not PRESETS:
    pytest.skip(
        "no advanced presets applicable for current encode_array build",
        allow_module_level=True,
    )


@pytest.mark.parametrize("img_path", IMAGE_PATHS, ids=lambda p: Path(p).name)
@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_integration_advanced_features(
    img_path: Path, mode: Literal["L", "RGB"], preset: Preset
) -> None:
    from fastfractal.core.decode import decode_array  # noqa: WPS433
    from fastfractal.core.encode import encode_array  # noqa: WPS433

    x = _load_img(img_path, mode=mode, max_side=160)

    kwargs = dict(preset.params)

    if "backend" in kwargs and _ALLOWED_BACKENDS:
        b = str(kwargs["backend"]).lower()
        if b not in _ALLOWED_BACKENDS:
            pytest.skip(
                f"backend '{b}' not supported; available={sorted(_ALLOWED_BACKENDS)}"
            )

    code = encode_array(x, **kwargs)
    rec = decode_array(code, iterations=int(preset.decode_iterations))

    if mode == "L" and rec.ndim == 3 and rec.shape[2] == 1:
        rec = rec[:, :, 0].astype(np.float32, copy=False)

    _assert_reconstruction(
        x.astype(np.float32, copy=False), rec.astype(np.float32, copy=False)
    )
