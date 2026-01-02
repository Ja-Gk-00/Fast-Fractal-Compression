from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("YAML root must be a mapping")
    return cast(dict[str, Any], obj)


def get_section(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    v = cfg.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"section {key} must be a mapping")
    return cast(dict[str, Any], v)


def _pick(cfg: Mapping[str, Any], key: str, alt_key: str | None) -> Any | None:
    if key in cfg:
        return cfg[key]
    if alt_key is not None and alt_key in cfg:
        return cfg[alt_key]
    return None


def pick_int(
    cfg: Mapping[str, Any], key: str, alt_key: str | None, default: int
) -> int:
    v = _pick(cfg, key, alt_key)
    if v is None:
        return int(default)
    if isinstance(v, bool):
        raise ValueError(f"{key} must be int")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(int(v)) == float(v):
        return int(v)
    raise ValueError(f"{key} must be int")


def pick_opt_int(
    cfg: Mapping[str, Any], key: str, alt_key: str | None, default: int | None
) -> int | None:
    v = _pick(cfg, key, alt_key)
    if v is None:
        return default
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"{key} must be int|null")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(int(v)) == float(v):
        return int(v)
    raise ValueError(f"{key} must be int|null")


def pick_float(
    cfg: Mapping[str, Any], key: str, alt_key: str | None, default: float
) -> float:
    v = _pick(cfg, key, alt_key)
    if v is None:
        return float(default)
    if isinstance(v, bool):
        raise ValueError(f"{key} must be float")
    if isinstance(v, (int, float)):
        return float(v)
    raise ValueError(f"{key} must be float")


def pick_bool(
    cfg: Mapping[str, Any], key: str, alt_key: str | None, default: bool
) -> bool:
    v = _pick(cfg, key, alt_key)
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    raise ValueError(f"{key} must be bool")


def pick_str(
    cfg: Mapping[str, Any], key: str, alt_key: str | None, default: str
) -> str:
    v = _pick(cfg, key, alt_key)
    if v is None:
        return str(default)
    if isinstance(v, str):
        return str(v)
    raise ValueError(f"{key} must be str")
