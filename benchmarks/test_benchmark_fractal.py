from __future__ import annotations

from io import BytesIO
from typing import Any, cast

import numpy as np
import pytest
from conftest import BenchItem, ImageSpec
from numpy.typing import NDArray

from fastfractal.core.decode import decode_array
from fastfractal.core.encode import encode_array
from fastfractal.io.codebook import dump_code, load_code_bytes


def _img_u8(img: NDArray[np.float32]) -> NDArray[np.uint8]:
    return np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)


def _img_u8_size(img: NDArray[np.float32]) -> int:
    return int(_img_u8(img).nbytes)


def _make_synthetic(
    size: int, channels: int, kind: str, seed: int
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    if kind == "random":
        if channels == 1:
            return rng.random((size, size), dtype=np.float32)
        return rng.random((size, size, 3), dtype=np.float32)

    if kind == "smooth":
        y = np.linspace(0.0, 1.0, size, dtype=np.float32)
        x = np.linspace(0.0, 1.0, size, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        base = 0.6 * yy + 0.4 * xx
        if channels == 1:
            return base.astype(np.float32, copy=False)
        out = np.stack(
            [
                base,
                np.clip(base * 0.9 + 0.05, 0.0, 1.0),
                np.clip(base * 0.8 + 0.1, 0.0, 1.0),
            ],
            axis=2,
        )
        return out.astype(np.float32, copy=False)

    if kind == "edges":
        img = np.zeros((size, size), dtype=np.float32)
        step = max(4, size // 16)
        for i in range(0, size, step):
            img[:, i : i + max(1, step // 2)] = 1.0
        if channels == 1:
            return img
        return np.stack(
            [
                img,
                np.roll(img, shift=step // 3, axis=0),
                np.roll(img, shift=step // 2, axis=1),
            ],
            axis=2,
        ).astype(np.float32, copy=False)

    raise ValueError("bad kind")


def _load_image_file(path: str, size: int, channels: int) -> NDArray[np.float32]:
    try:
        from PIL import Image
    except Exception:
        pytest.skip("Pillow is required to load file images for benchmarks")

    with Image.open(path) as im:
        if channels == 1:
            im2 = im.convert("L")
        else:
            im2 = im.convert("RGB")
        im3 = im2.resize((size, size))
        arr = np.asarray(im3, dtype=np.uint8)
    if channels == 1:
        return cast(NDArray[np.float32], arr.astype(np.float32) / 255.0)
    return cast(NDArray[np.float32], arr.astype(np.float32) / 255.0)


def _get_image(imgspec: ImageSpec, bench_size: int) -> NDArray[np.float32]:
    size = int(imgspec.size) if imgspec.size is not None else int(bench_size)
    if imgspec.kind == "synthetic":
        return _make_synthetic(
            size=size,
            channels=int(imgspec.channels),
            kind=imgspec.pattern,
            seed=int(imgspec.seed),
        )
    if imgspec.kind == "file":
        if imgspec.path is None:
            raise ValueError("file image missing path")
        ch = int(imgspec.channels)
        if ch == 0:
            ch = 3
        return _load_image_file(str(imgspec.path), size=size, channels=ch)
    raise ValueError("bad image kind")


def _jpeg_encode_bytes(img: NDArray[np.float32], quality: int) -> bytes:
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Pillow is required for JPEG benchmarks") from ImportError

    u8 = _img_u8(img)
    if img.ndim == 2:
        pil = Image.fromarray(u8, mode="L")
    else:
        pil = Image.fromarray(u8, mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def _jpeg_decode_bytes(data: bytes, channels: int) -> NDArray[np.float32]:
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Pillow is required for JPEG benchmarks") from ImportError

    buf = BytesIO(data)
    with Image.open(buf) as im:
        if channels == 1:
            im2 = im.convert("L")
        else:
            im2 = im.convert("RGB")
        arr = np.asarray(im2, dtype=np.uint8)
    if channels == 1:
        return cast(NDArray[np.float32], arr.astype(np.float32) / 255.0)
    return cast(NDArray[np.float32], arr.astype(np.float32) / 255.0)


@pytest.fixture(scope="session")
def bench_rounds(pytestconfig: pytest.Config) -> int:
    return int(pytestconfig.getoption("--bench-rounds"))


@pytest.fixture(scope="session")
def bench_warmup_rounds(pytestconfig: pytest.Config) -> int:
    return int(pytestconfig.getoption("--bench-warmup-rounds"))


@pytest.fixture(scope="session")
def bench_size(pytestconfig: pytest.Config) -> int:
    return int(pytestconfig.getoption("--bench-size"))


def _mode_allows(mode: str, what: str) -> bool:
    if mode == "both":
        return True
    return mode == what


def test_bench_ffc_encode_speed(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if not _mode_allows(bench_item.mode, "encode"):
        pytest.skip("encode disabled for this case")

    img = _get_image(bench_item.image, bench_size=bench_size)
    orig_bytes = _img_u8_size(img)
    code0 = encode_array(img, **bench_item.encode_kwargs)
    ffc_bytes0 = int(len(dump_code(code0)))
    cr0 = float(orig_bytes) / float(max(1, ffc_bytes0))

    benchmark.extra_info["orig_bytes"] = int(orig_bytes)
    benchmark.extra_info["ffc_bytes"] = int(ffc_bytes0)
    benchmark.extra_info["ffc_compression_ratio"] = float(cr0)

    def run() -> object:
        return encode_array(img, **bench_item.encode_kwargs)

    benchmark.pedantic(
        run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
    )


def test_bench_ffc_decode_speed(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if not _mode_allows(bench_item.mode, "decode"):
        pytest.skip("decode disabled for this case")

    img = _get_image(bench_item.image, bench_size=bench_size)
    orig_bytes = _img_u8_size(img)
    code = encode_array(img, **bench_item.encode_kwargs)
    data = dump_code(code)
    ffc_bytes0 = int(len(data))
    cr0 = float(orig_bytes) / float(max(1, ffc_bytes0))

    benchmark.extra_info["orig_bytes"] = int(orig_bytes)
    benchmark.extra_info["ffc_bytes"] = int(ffc_bytes0)
    benchmark.extra_info["ffc_compression_ratio"] = float(cr0)

    iters = int(bench_item.decode_kwargs.get("iterations", 8))

    def run() -> object:
        return decode_array(code, iterations=iters)

    benchmark.pedantic(
        run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
    )


def test_bench_ffc_roundtrip_speed(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if bench_item.mode != "both":
        pytest.skip("roundtrip only for mode=both")

    img = _get_image(bench_item.image, bench_size=bench_size)
    iters = int(bench_item.decode_kwargs.get("iterations", 8))

    def run() -> object:
        code = encode_array(img, **bench_item.encode_kwargs)
        return decode_array(code, iterations=iters)

    benchmark.pedantic(
        run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
    )


def test_bench_ffc_compression_rate_and_serialize(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if not _mode_allows(bench_item.mode, "encode"):
        pytest.skip("encode disabled for this case")

    img = _get_image(bench_item.image, bench_size=bench_size)
    orig_bytes = _img_u8_size(img)

    def run() -> int:
        code = encode_array(img, **bench_item.encode_kwargs)
        b = dump_code(code)
        return int(len(b))

    code_bytes = int(
        benchmark.pedantic(
            run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
        )
    )
    cr = float(orig_bytes) / float(max(1, code_bytes))
    benchmark.extra_info["orig_bytes"] = int(orig_bytes)
    benchmark.extra_info["ffc_bytes"] = int(code_bytes)
    benchmark.extra_info["ffc_compression_ratio"] = float(cr)


def test_bench_ffc_decode_from_bytes_speed(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if not _mode_allows(bench_item.mode, "decode"):
        pytest.skip("decode disabled for this case")

    img = _get_image(bench_item.image, bench_size=bench_size)
    code = encode_array(img, **bench_item.encode_kwargs)
    data = dump_code(code)
    iters = int(bench_item.decode_kwargs.get("iterations", 8))

    def run() -> object:
        c2 = load_code_bytes(data)
        return decode_array(c2, iterations=iters)

    benchmark.pedantic(
        run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
    )


def test_bench_jpeg_encode_speed(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if not bench_item.include_jpeg:
        pytest.skip("jpeg disabled")
    if not _mode_allows(bench_item.mode, "encode"):
        pytest.skip("encode disabled for this case")

    img = _get_image(bench_item.image, bench_size=bench_size)
    orig_bytes = _img_u8_size(img)
    q = int(bench_item.jpeg_quality)

    try:
        data0 = _jpeg_encode_bytes(img, q)
    except Exception:
        pytest.skip("Pillow missing for JPEG benchmarks")

    jpeg_bytes0 = int(len(data0))
    cr0 = float(orig_bytes) / float(max(1, jpeg_bytes0))

    benchmark.extra_info["orig_bytes"] = int(orig_bytes)
    benchmark.extra_info["jpeg_bytes"] = int(jpeg_bytes0)
    benchmark.extra_info["jpeg_compression_ratio"] = float(cr0)
    benchmark.extra_info["jpeg_quality"] = int(q)

    def run() -> int:
        return int(len(_jpeg_encode_bytes(img, q)))

    benchmark.pedantic(
        run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
    )


def test_bench_jpeg_decode_speed(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if not bench_item.include_jpeg:
        pytest.skip("jpeg disabled")
    if not _mode_allows(bench_item.mode, "decode"):
        pytest.skip("decode disabled for this case")

    img = _get_image(bench_item.image, bench_size=bench_size)
    ch = 1 if img.ndim == 2 else 3
    q = int(bench_item.jpeg_quality)

    try:
        data = _jpeg_encode_bytes(img, q)
    except Exception:
        pytest.skip("Pillow missing for JPEG benchmarks")

    def run() -> object:
        return _jpeg_decode_bytes(data, channels=ch)

    benchmark.pedantic(
        run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
    )


def test_bench_jpeg_compression_rate(
    benchmark: Any,
    bench_item: BenchItem,
    bench_size: int,
    bench_rounds: int,
    bench_warmup_rounds: int,
) -> None:
    if not bench_item.include_jpeg:
        pytest.skip("jpeg disabled")
    if not _mode_allows(bench_item.mode, "encode"):
        pytest.skip("encode disabled for this case")

    img = _get_image(bench_item.image, bench_size=bench_size)
    orig_bytes = _img_u8_size(img)
    q = int(bench_item.jpeg_quality)

    try:
        _jpeg_encode_bytes(img, q)
    except Exception:
        pytest.skip("Pillow missing for JPEG benchmarks")

    def run() -> int:
        return int(len(_jpeg_encode_bytes(img, q)))

    jpeg_bytes = int(
        benchmark.pedantic(
            run, rounds=bench_rounds, warmup_rounds=bench_warmup_rounds, iterations=1
        )
    )
    cr = float(orig_bytes) / float(max(1, jpeg_bytes))
    benchmark.extra_info["orig_bytes"] = int(orig_bytes)
    benchmark.extra_info["jpeg_bytes"] = int(jpeg_bytes)
    benchmark.extra_info["jpeg_compression_ratio"] = float(cr)
    benchmark.extra_info["jpeg_quality"] = int(q)
