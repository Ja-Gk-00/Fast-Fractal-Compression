try:
    from . import _cext
except ImportError:
    _cext = None
