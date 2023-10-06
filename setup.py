import sys
import os
from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension

import distutils.sysconfig

__version__ = "0.0.1"

if (sys.platform[:6] == "darwin"
        and ("clang" in distutils.sysconfig.get_config_var("CC")
                or "clang" in os.environ.get("CC", ""))):
    compiler_args = ["-Xpreprocessor"]
    linker_args = ["-mlinker-version=305", "-Xpreprocessor"]
else:
    compiler_args = []
    linker_args = []

compiler_args += ["-fopenmp"]
linker_args += ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "levin",
        ["./LevinBessel/src/levin.cpp", "./LevinBessel/python/pybind11_interface.cpp"],
        cxx_std=11,
        include_dirs=["./LevinBessel/src"],
        libraries=["m", "gsl", "gslcblas"],
        extra_compile_args=compiler_args,
        extra_link_args=linker_args
        ),
]

setup(
    name="levin",
    version=__version__,
    # author="Robert Reischke",
    # author_email="s",
    # url="",
    # description="",
    # long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    install_requires=[
        'future',
        'pyyaml',
        'numpy',
        'astropy',
        'healpy',
        'pip>=20.0',
        'gsl',
        'scipy',
        'wget',
        'matplotlib',
        'hmf']
)