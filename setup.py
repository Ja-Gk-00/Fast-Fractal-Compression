from __future__ import annotations

from pathlib import Path

import numpy as np
from setuptools import Extension, setup, find_packages

from mypyc.build import mypycify

root = Path(__file__).parent

cext = Extension(
    "fastfractal._cext",
    sources=["fastfractal/_cext.c", "fastfractal/cext_encode_leaf.c"],
    include_dirs=[np.get_include()],
)

mypyc_targets = [
    "fastfractal/utils/entropy.py",
    "fastfractal/utils/metrics.py",
    "fastfractal/core/decode.py",
    #"fastfractal/core/encode.py", # uses C wrappers and should not be compiled
    "fastfractal/core/transforms.py",
    #"fastfractal/core/search.py", #TODO revamp so mypyc can compile
    "fastfractal/core/blocks.py",
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
    ext_modules=ext_modules,
    zip_safe=False,
)
