import distutils.sysconfig
import os
import sys

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "1.0.0"


def _fix_missing_compilers():
    """Fall back to a compiler on PATH if the configured one (e.g. a conda
    cross-compiler such as x86_64-conda-linux-gnu-c++) is not installed."""
    import shutil

    def first_word(cmd):
        return cmd.split()[0] if cmd and cmd.split() else ""

    for var, candidates in (("CC", ("gcc", "cc", "clang")),
                            ("CXX", ("g++", "c++", "clang++"))):
        configured = os.environ.get(var) or distutils.sysconfig.get_config_var(var) or ""
        exe = first_word(configured)
        if exe and shutil.which(exe):
            continue
        for cand in candidates:
            path = shutil.which(cand)
            if path:
                print(f"WARNING: compiler '{exe or var}' not found, using '{path}' for {var}. "
                      "Install gxx_linux-64 from conda-forge to use the conda toolchain.")
                os.environ[var] = path
                break


_fix_missing_compilers()

if (sys.platform[:6] == "darwin"
        and ("clang" in distutils.sysconfig.get_config_var("CC")
                or "clang" in os.environ.get("CC", ""))):
    compiler_args = ["-Xpreprocessor"]
    linker_args = ["-mlinker-version=305", "-Xpreprocessor"]
    # "-Xpreprocessor -fopenmp" does not link the OpenMP runtime, link libomp explicitly
    omp_link_args = ["-lomp"]
else:
    compiler_args = ["-shared"]
    linker_args = ["-shared"]
    omp_link_args = []

compiler_args_levin =  compiler_args + ["-fopenmp", "-O3"]
linker_args_levin   =  linker_args   + ["-fopenmp", "-O3"] + omp_link_args

compiler = "g++"
compiler_args_discretecov =  compiler_args + ["-fopenmp", "-O3"]
linker_args_discretecov   =  linker_args   + ["-fopenmp", "-O3"]
ext_modules = [
    Pybind11Extension(
        "levin",
        ["./LevinBessel/src/levin.cpp", "./LevinBessel/python/pybind11_interface.cpp"],
        cxx_std=14,
        include_dirs=["./LevinBessel/src"],
        libraries=["m", "gsl", "gslcblas"],
        extra_compile_args=compiler_args_levin,
        extra_link_args=linker_args_levin
        )
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
        'numpy<1.24',
        'astropy',
        'healpy',
        'pip>=20.0',
        'gsl',
        'scipy>=1.12.0',  # scipy.integrate.cumulative_simpson
        'wget',
        'matplotlib',
        'hmf',
        'camb',
        'mpmath']
)
