from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(s: str) -> object:
    t = s.strip()
    if t == "":
        return ""
    if t in ("null", "Null", "NULL", "none", "None", "NONE", "~"):
        return None
    if t in ("true", "True", "TRUE"):
        return True
    if t in ("false", "False", "FALSE"):
        return False
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if inner == "":
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [_parse_scalar(p) for p in parts]
    try:
        if t.startswith("0") and len(t) > 1 and t[1].isdigit():
            raise ValueError
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def load_yaml(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object] | list[object]]] = [(0, root)]

    def cur_container() -> dict[str, object] | list[object]:
        return stack[-1][1]

    def pop_to(indent: int) -> None:
        while len(stack) > 1 and stack[-1][0] > indent:
            stack.pop()

    last_key_stack: list[str] = []

    for raw in lines:
        if raw.strip() == "":
            continue
        s = raw.rstrip("\n")
        ls = s.lstrip(" ")
        if ls.startswith("#"):
            continue
        indent = len(s) - len(ls)
        if indent % 2 != 0:
            raise ValueError("invalid indentation")
        pop_to(indent)
        cont = cur_container()

        if ls.startswith("- "):
            item_str = ls[2:].strip()
            if not isinstance(cont, list):
                if not last_key_stack:
                    raise ValueError("list item without a list parent")
                pop_to(indent)
                cont2 = cur_container()
                if not isinstance(cont2, dict):
                    raise ValueError("invalid parent for list promotion")
                key = last_key_stack[-1]
                new_list: list[object] = []
                cont2[key] = new_list
                stack.append((indent, new_list))
                cont = new_list

            if item_str == "":
                cont.append({})
                stack.append((indent + 2, cont[-1]))
            else:
                if ":" in item_str:
                    k, v = item_str.split(":", 1)
                    k2 = k.strip()
                    v2 = v.strip()
                    d: dict[str, object] = {}
                    if v2 == "":
                        d[k2] = {}
                        cont.append(d)
                        last_key_stack.append(k2)
                        stack.append((indent + 2, d))
                        stack.append((indent + 4, d[k2]))
                    else:
                        d[k2] = _parse_scalar(v2)
                        cont.append(d)
                        last_key_stack.append(k2)
                else:
                    cont.append(_parse_scalar(item_str))
            continue

        if ":" not in ls:
            raise ValueError("invalid line")

        key, rest = ls.split(":", 1)
        k = key.strip()
        v = rest.strip()

        if not isinstance(cont, dict):
            raise ValueError("mapping entry under list without dict item")

        last_key_stack.append(k)

        if v == "":
            nxt: dict[str, object] = {}
            cont[k] = nxt
            stack.append((indent + 2, nxt))
            continue

        cont[k] = _parse_scalar(v)

    return root


def get_section(cfg: dict[str, object], name: str) -> dict[str, object]:
    v = cfg.get(name)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError("config section must be a mapping")
    return v


def pick_bool(cfg: dict[str, object], key: str, cli: bool | None, default: bool) -> bool:
    if cli is not None:
        return bool(cli)
    v = cfg.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        if v in ("true", "True", "TRUE", "1", "yes", "Yes", "YES", "on", "On", "ON"):
            return True
        if v in ("false", "False", "FALSE", "0", "no", "No", "NO", "off", "Off", "OFF"):
            return False
    raise ValueError("invalid bool in config")


def pick_int(cfg: dict[str, object], key: str, cli: int | None, default: int) -> int:
    if cli is not None:
        return int(cli)
    v = cfg.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        return int(v)
    raise ValueError("invalid int in config")


def pick_float(cfg: dict[str, object], key: str, cli: float | None, default: float) -> float:
    if cli is not None:
        return float(cli)
    v = cfg.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(v)
    raise ValueError("invalid float in config")


def pick_str(cfg: dict[str, object], key: str, cli: str | None, default: str) -> str:
    if cli is not None:
        return str(cli)
    v = cfg.get(key)
    if v is None:
        return default
    if isinstance(v, str):
        return v
    raise ValueError("invalid str in config")


def pick_opt_int(cfg: dict[str, object], key: str, cli: int | None, default: int | None) -> int | None:
    if cli is not None:
        return None if int(cli) <= 0 else int(cli)
    v = cfg.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return None if v <= 0 else v
    if isinstance(v, float):
        vv = int(v)
        return None if vv <= 0 else vv
    if isinstance(v, str):
        vv = int(v)
        return None if vv <= 0 else vv
    raise ValueError("invalid optional int in config")


def ensure_mapping(x: object) -> dict[str, object]:
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise ValueError("expected mapping")
    return x


def as_dict_any(x: dict[str, object]) -> dict[str, Any]:
    return x  # type: ignore[return-value]
