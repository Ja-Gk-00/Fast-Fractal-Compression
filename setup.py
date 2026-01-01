from __future__ import annotations

from pathlib import Path

import numpy as np
from setuptools import Extension, setup, find_packages

root = Path(__file__).parent

ext = Extension(
    "fastfractal._cext",
    sources=[str("fastfractal" + "/_cext.c")],
    include_dirs=[np.get_include()],
)

setup(
    packages=find_packages(where="."),
    ext_modules=[ext],
)
