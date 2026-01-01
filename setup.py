from __future__ import annotations

from pathlib import Path

import numpy as np
from setuptools import Extension, setup, find_packages

from mypyc.build import mypycify

root = Path(__file__).parent

cext = Extension(
    "fastfractal._cext",
    sources=["fastfractal/_cext.c"],
    include_dirs=[np.get_include()],
)

mypyc_targets = [
    "fastfractal/utils/entropy.py",
]

ext_modules = [
    cext,
    *mypycify(
        mypyc_targets,
        debug_level="0",
    ),
]

setup(
    packages=find_packages(where="."),
    ext_modules=cext,
    zip_safe=False,
)
