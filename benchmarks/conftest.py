from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, cast

import pytest

from fastfractal.io.config import (
    get_section,
    load_yaml,
    pick_bool,
    pick_float,
    pick_int,
    pick_opt_int,
    pick_str,
)


@dataclass(frozen=True, slots=True)
class ImageSpec:
    name: str
    kind: str
    path: Path | None
    pattern: str
    channels: int
    size: int | None
    seed: int


@dataclass(frozen=True, slots=True)
class BenchItem:
    name: str
    encode_kwargs: dict[str, Any]
    decode_kwargs: dict[str, Any]
    mode: str
    image: ImageSpec
    jpeg_quality: int
    include_jpeg: bool


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("bench")
    g.addoption(
        "--bench-suite",
        action="append",
        default=[],
        help="Benchmark suite YAML file(s). Can be repeated.",
    )
    g.addoption(
        "--bench-config",
        action="append",
        default=[],
        help="Codec YAML config file(s). Can be repeated.",
    )
    g.addoption("--bench-size", action="store", type=int, default=128)
    g.addoption("--bench-rounds", action="store", type=int, default=10)
    g.addoption("--bench-warmup-rounds", action="store", type=int, default=3)
    g.addoption(
        "--bench-case",
        action="append",
        default=[],
        help="Run only selected benchmark case name(s). Can be repeated.",
    )
    g.addoption(
        "--bench-case-range",
        action="store",
        type=str,
        default="",
        help="Slice of cases, e.g. 0:5, 3:, :10",
    )
    g.addoption(
        "--bench-no-jpeg",
        action="store_true",
        default=False,
        help="Disable JPEG reference benchmarks.",
    )
    g.addoption(
        "--bench-case-prefix",
        action="append",
        default=[],
        help="Run only cases whose expanded id starts with the given prefix. Can be repeated.",
    )
    g.addoption("--bench-out", action="store", type=str, default="benchmarks/out")
    g.addoption("--bench-no-save", action="store_true", default=False)


def pytest_configure(config: pytest.Config) -> None:
    if bool(getattr(config.option, "bench_no_save", False)):
        return

    if hasattr(config.option, "benchmark_autosave"):
        config.option.benchmark_autosave = True

    out_dir = Path(str(getattr(config.option, "bench_out", "benchmarks/out"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(config.option, "benchmark_json"):
        cur = getattr(config.option, "benchmark_json", None)

        if cur is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = out_dir / f"bench_{ts}.json"
            config.option.benchmark_json = p.open("wb")
            return

        if hasattr(cur, "write"):
            return

        if isinstance(cur, str) and cur.strip() != "":
            p2 = Path(cur).resolve()
            p2.parent.mkdir(parents=True, exist_ok=True)
            config.option.benchmark_json = p2.open("wb")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p3 = out_dir / f"bench_{ts}.json"
        config.option.benchmark_json = p3.open("wb")


def _as_mapping(x: object) -> dict[str, object]:
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise ValueError("expected mapping")
    return x


def _as_list(x: object) -> list[object]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    raise ValueError("expected list")


def _range_values(start: float, stop: float, step: float) -> list[float]:
    if step == 0.0:
        raise ValueError("range step must be non-zero")
    out: list[float] = []
    eps = abs(step) * 1e-12 + 1e-12
    if step > 0:
        v = start
        while v <= stop + eps:
            out.append(v)
            v += step
    else:
        v = start
        while v >= stop - eps:
            out.append(v)
            v += step
    return out


def _values_from_spec(v: object) -> list[object]:
    if isinstance(v, list):
        return list(v)
    if isinstance(v, dict):
        m = cast(dict[str, object], v)
        if "start" in m and "stop" in m and "step" in m:
            s0 = float(cast(object, m["start"]))
            s1 = float(cast(object, m["stop"]))
            st = float(cast(object, m["step"]))
            vals = _range_values(s0, s1, st)
            if all(float(int(x)) == x for x in vals):
                return [int(x) for x in vals]
            return vals
        return [m]
    return [v]


def _expand_overrides(
    base: dict[str, Any], overrides: dict[str, object]
) -> list[dict[str, Any]]:
    if len(overrides) == 0:
        return [dict(base)]
    keys = sorted(overrides.keys())
    value_lists = [_values_from_spec(overrides[k]) for k in keys]
    out: list[dict[str, Any]] = []
    for combo in product(*value_lists):
        d = dict(base)
        for k, v in zip(keys, combo, strict=False):
            d[k] = v
        out.append(d)
    return out


def _default_encode_kwargs() -> dict[str, Any]:
    return {
        "min_block": 4,
        "max_block": 16,
        "stride": 4,
        "use_quadtree": False,
        "max_mse": 0.0025,
        "use_buckets": False,
        "bucket_count": 1,
        "use_s_sets": False,
        "topk": 64,
        "backend": "dot",
        "lsh_budget": 2048,
        "entropy_thresh": 0.0,
        "quantized": False,
        "s_clip": 0.99,
        "o_min": -0.5,
        "o_max": 1.5,
        "pca_dim": 16,
        "lsh_planes": 16,
        "seed": 0,
        "max_domains": None,
        "block": None,
        "iterations_hint": 8,
    }


def _default_decode_kwargs() -> dict[str, Any]:
    return {"iterations": 8}


def _resolve_codec_from_cfg(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg_all = load_yaml(path)
    enc = get_section(cfg_all, "encode")
    dec = get_section(cfg_all, "decode")

    max_block = pick_int(enc, "max_block", None, 16)
    min_block = pick_int(enc, "min_block", None, 4)
    stride = pick_int(enc, "stride", None, 4)

    use_quadtree = pick_bool(enc, "quadtree", None, False)
    max_mse = pick_float(enc, "max_mse", None, 0.0025)

    buckets = pick_int(enc, "buckets", None, 0)
    use_buckets = buckets > 0
    bucket_count = buckets if use_buckets else 1

    use_s_sets = pick_bool(enc, "s_sets", None, False)

    quantized = pick_bool(enc, "quantize", None, False)
    s_clip = pick_float(enc, "s_clip", None, 0.99)
    o_min = pick_float(enc, "o_min", None, -0.5)
    o_max = pick_float(enc, "o_max", None, 1.5)

    backend = pick_str(enc, "backend", None, "dot")
    topk = pick_int(enc, "topk", None, 64)
    lsh_budget = pick_int(enc, "lsh_budget", None, 2048)
    pca_dim = pick_int(enc, "pca_dim", None, 16)
    lsh_planes = pick_int(enc, "lsh_planes", None, 16)
    seed = pick_int(enc, "seed", None, 0)

    entropy_thresh = pick_float(enc, "entropy_thresh", None, 0.0)
    max_domains = pick_opt_int(enc, "max_domains", None, None)

    iters = pick_int(dec, "iters", None, 8)

    encode_kwargs: dict[str, Any] = _default_encode_kwargs()
    encode_kwargs.update(
        {
            "min_block": min_block,
            "max_block": max_block,
            "stride": stride,
            "use_quadtree": use_quadtree,
            "max_mse": max_mse,
            "use_buckets": use_buckets,
            "bucket_count": bucket_count,
            "use_s_sets": use_s_sets,
            "topk": topk,
            "backend": backend,
            "lsh_budget": lsh_budget,
            "entropy_thresh": entropy_thresh,
            "quantized": quantized,
            "s_clip": s_clip,
            "o_min": o_min,
            "o_max": o_max,
            "pca_dim": pca_dim,
            "lsh_planes": lsh_planes,
            "seed": seed,
            "max_domains": max_domains,
        }
    )

    decode_kwargs: dict[str, Any] = _default_decode_kwargs()
    decode_kwargs.update({"iterations": iters})

    return encode_kwargs, decode_kwargs


def _images_from_case(case: dict[str, object], base_dir: Path) -> list[ImageSpec]:
    if "images" in case:
        imgs = _as_list(case["images"])
        out: list[ImageSpec] = []
        for i, x in enumerate(imgs):
            m = _as_mapping(x)
            nm = str(m.get("name") or f"img{i}")
            p = m.get("path") or m.get("image")
            if p is None:
                raise ValueError("image entry missing path")
            path = (base_dir / str(p)).resolve()
            ch = int(m.get("channels") or 0)
            if ch not in (0, 1, 3):
                raise ValueError("channels must be 1 or 3")
            size = m.get("size")
            size_i = int(size) if size is not None else None
            out.append(
                ImageSpec(
                    name=nm,
                    kind="file",
                    path=path,
                    pattern="",
                    channels=ch if ch != 0 else 0,
                    size=size_i,
                    seed=0,
                )
            )
        return out

    if "image" in case or "path" in case:
        p = case.get("image") or case.get("path")
        if p is None:
            raise ValueError("image missing path")
        path = (base_dir / str(p)).resolve()
        ch0 = int(case.get("channels") or 0)
        if ch0 not in (0, 1, 3):
            raise ValueError("channels must be 1 or 3")
        size0 = case.get("size")
        size_i0 = int(size0) if size0 is not None else None
        return [
            ImageSpec(
                name=str(case.get("image_name") or Path(str(p)).stem),
                kind="file",
                path=path,
                pattern="",
                channels=ch0,
                size=size_i0,
                seed=0,
            )
        ]

    pat = str(case.get("pattern") or "random")
    ch = int(case.get("channels") or 3)
    if ch not in (1, 3):
        raise ValueError("synthetic channels must be 1 or 3")
    size0 = case.get("size")
    size_i0 = int(size0) if size0 is not None else None
    seed = int(case.get("seed") or 0)
    return [
        ImageSpec(
            name=f"syn_{pat}_{ch}c",
            kind="synthetic",
            path=None,
            pattern=pat,
            channels=ch,
            size=size_i0,
            seed=seed,
        )
    ]


def _summarize_overrides(d: dict[str, Any], keys: Sequence[str]) -> str:
    parts: list[str] = []
    for k in keys:
        if k in d:
            v = d[k]
            if v is None:
                continue
            parts.append(f"{k}={v}")
    return "__".join(parts)


def _load_suite(path: Path, include_jpeg_cli: bool) -> list[BenchItem]:
    cfg = load_yaml(path)
    bench = _as_mapping(cfg.get("bench"))
    defaults = _as_mapping(bench.get("defaults"))
    cases = _as_list(bench.get("cases"))
    max_items = int(bench.get("max_items") or 512)

    out: list[BenchItem] = []
    base_dir = path.parent

    for i, c0 in enumerate(cases):
        case = _as_mapping(c0)
        base_name = str(case.get("name") or f"{path.stem}_{i}")
        mode = str(case.get("mode") or defaults.get("mode") or "both")
        if mode not in ("encode", "decode", "both"):
            raise ValueError("mode must be encode/decode/both")

        jpeg_quality = int(
            case.get("jpeg_quality") or defaults.get("jpeg_quality") or 90
        )

        include_jpeg = bool(
            case.get("include_jpeg")
            if "include_jpeg" in case
            else bench.get("include_jpeg")
        )
        if "include_jpeg" not in case and "include_jpeg" not in bench:
            include_jpeg = True
        if include_jpeg_cli is False:
            include_jpeg = False

        if "codec_config" in case or "config" in case:
            p = case.get("codec_config") or case.get("config")
            if p is None:
                raise ValueError("codec_config missing")
            enc0, dec0 = _resolve_codec_from_cfg((base_dir / str(p)).resolve())
        else:
            enc0 = _default_encode_kwargs()
            dec0 = _default_decode_kwargs()

        enc_over = _as_mapping(case.get("encode"))
        dec_over = _as_mapping(case.get("decode"))

        enc_variants = _expand_overrides(enc0, enc_over)
        dec_variants = _expand_overrides(dec0, dec_over)

        imgs = _images_from_case(case, base_dir)

        key_hint_enc = sorted(enc_over.keys())
        key_hint_dec = sorted(dec_over.keys())

        for enc_k in enc_variants:
            for dec_k in dec_variants:
                enc_tag = _summarize_overrides(enc_k, key_hint_enc)
                dec_tag = _summarize_overrides(dec_k, key_hint_dec)
                for im in imgs:
                    nm_parts: list[str] = [base_name]
                    if im.name:
                        nm_parts.append(im.name)
                    if enc_tag:
                        nm_parts.append(enc_tag)
                    if dec_tag:
                        nm_parts.append(dec_tag)
                    nm = "__".join(nm_parts)

                    out.append(
                        BenchItem(
                            name=nm,
                            encode_kwargs=enc_k,
                            decode_kwargs=dec_k,
                            mode=mode,
                            image=im,
                            jpeg_quality=jpeg_quality,
                            include_jpeg=include_jpeg,
                        )
                    )

                    if len(out) >= max_items:
                        return out
    return out


def _builtin_items(size: int, include_jpeg: bool) -> list[BenchItem]:
    base_enc = _default_encode_kwargs()
    base_enc.update(
        {
            "min_block": 16,
            "max_block": 16,
            "stride": 8,
            "use_quadtree": False,
            "topk": 32,
            "backend": "dot",
            "max_domains": 3000,
        }
    )
    base_dec = {"iterations": 8}

    patterns = ["random", "smooth", "edges"]
    chans = [1, 3]
    out: list[BenchItem] = []
    for pat in patterns:
        for ch in chans:
            im = ImageSpec(
                name=f"syn_{pat}_{ch}c_{size}",
                kind="synthetic",
                path=None,
                pattern=pat,
                channels=ch,
                size=size,
                seed=123,
            )
            out.append(
                BenchItem(
                    name=f"baseline_dot__{im.name}",
                    encode_kwargs=dict(base_enc),
                    decode_kwargs=dict(base_dec),
                    mode="both",
                    image=im,
                    jpeg_quality=90,
                    include_jpeg=include_jpeg,
                )
            )
    return out


def _apply_selection(
    items: list[BenchItem], only_names: list[str], prefixes: list[str], case_range: str
) -> list[BenchItem]:
    out = list(items)

    if len(only_names) > 0:
        s = set(only_names)
        out = [x for x in out if x.name in s]

    if len(prefixes) > 0:
        out = [x for x in out if any(x.name.startswith(p) for p in prefixes)]

    if case_range.strip() != "":
        txt = case_range.strip()
        if ":" not in txt:
            raise ValueError("bench-case-range must be like start:end")
        a, b = txt.split(":", 1)
        start = int(a) if a != "" else None
        end = int(b) if b != "" else None
        out = out[slice(start, end)]

    return out


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "bench_item" not in metafunc.fixturenames:
        return

    cfg_paths = list(metafunc.config.getoption("--bench-config"))
    suite_paths = list(metafunc.config.getoption("--bench-suite"))
    size = int(metafunc.config.getoption("--bench-size"))
    include_jpeg = not bool(metafunc.config.getoption("--bench-no-jpeg"))

    items: list[BenchItem] = []

    for sp in suite_paths:
        items.extend(_load_suite(Path(sp), include_jpeg_cli=include_jpeg))

    if len(cfg_paths) > 0:
        for p in cfg_paths:
            enc, dec = _resolve_codec_from_cfg(Path(p))
            for pat in ("random", "smooth", "edges"):
                for ch in (1, 3):
                    im = ImageSpec(
                        name=f"syn_{pat}_{ch}c_{size}",
                        kind="synthetic",
                        path=None,
                        pattern=pat,
                        channels=ch,
                        size=size,
                        seed=123,
                    )
                    items.append(
                        BenchItem(
                            name=f"{Path(p).stem}__{im.name}",
                            encode_kwargs=enc,
                            decode_kwargs=dec,
                            mode="both",
                            image=im,
                            jpeg_quality=90,
                            include_jpeg=include_jpeg,
                        )
                    )

    if len(items) == 0:
        items = _builtin_items(size=size, include_jpeg=include_jpeg)

    only_names = list(metafunc.config.getoption("--bench-case"))
    prefixes = list(metafunc.config.getoption("--bench-case-prefix"))
    case_range = str(metafunc.config.getoption("--bench-case-range") or "")
    items = _apply_selection(
        items, only_names=only_names, prefixes=prefixes, case_range=case_range
    )

    metafunc.parametrize("bench_item", items, ids=[x.name for x in items])
