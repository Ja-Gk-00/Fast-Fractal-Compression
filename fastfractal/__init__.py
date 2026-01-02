try:
    from . import _cext  # type: ignore[attr-defined]
except ImportError:
    _cext = None
