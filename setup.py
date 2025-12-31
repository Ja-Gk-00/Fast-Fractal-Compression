# setup.py
from __future__ import annotations

import sys
from pathlib import Path

from setuptools import Extension, setup

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy is required to build fastfractal._cext") from e


def collect_c_sources() -> list[str]:
    csrc = Path("fastfractal") / "csrc"
    sources = [
        str(csrc / "_cextmodule.c"),
        str(csrc / "domains_yx.c"),
        str(csrc / "ranges_yx.c"),
        str(csrc / "extract_range.c"),
        str(csrc / "extract_range_flat.c"),
        str(csrc / "topk_dot.c"),
        str(csrc / "topk_from_subset.c"),
    ]
    return sources


extra_compile_args: list[str] = []
if sys.platform.startswith("win"):
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3"]

ext = Extension(
    "fastfractal._cext",
    sources=collect_c_sources(),
    include_dirs=[np.get_include(), "fastfractal/csrc"],
    extra_compile_args=extra_compile_args,
)

setup(ext_modules=[ext])
