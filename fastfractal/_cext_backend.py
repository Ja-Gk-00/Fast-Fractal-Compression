from __future__ import annotations

import os
from typing import Any


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if v in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    return default


class CExtBackend:
    def __init__(self) -> None:
        self._enabled_requested: bool = _env_flag("FASTFRACTAL_USE_C", True)
        self._mod = None
        self._import_error: Exception | None = None

        if self._enabled_requested:
            try:
                from . import _cext as _m
            except Exception as e:
                self._import_error = e
                self._mod = None
            else:
                self._mod = _m

    @property
    def enabled(self) -> bool:
        return self._enabled_requested and (self._mod is not None)

    def set_enabled(self, enabled: bool) -> None:
        """Optional runtime override (rarely needed)."""
        self._enabled_requested = bool(enabled)
        if self._enabled_requested and self._mod is None:
            try:
                from . import _cext as _m
            except Exception as e:
                self._import_error = e
                self._mod = None
            else:
                self._import_error = None
                self._mod = _m

    def import_error(self) -> Exception | None:
        return self._import_error

    def has(self, name: str) -> bool:
        return self.enabled and hasattr(self._mod, name)

    def get(self, name: str) -> Any:
        if not self.enabled:
            raise AttributeError(
                f"C extensions disabled/unavailable (FASTFRACTAL_USE_C={os.getenv('FASTFRACTAL_USE_C')!r}). "
                f"Missing attribute: {name}"
            )
        return getattr(self._mod, name)

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        fn = self.get(name)
        return fn(*args, **kwargs)


cext = CExtBackend()


def has(name: str) -> bool:
    return cext.has(name)


def get(name: str) -> Any:
    return cext.get(name)
