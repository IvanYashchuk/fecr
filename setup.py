import sys

from setuptools import setup

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

version = "0.1"

setup(
    name="fecr",
    description="Finite element chain rules (fecr): Firedrake/FEniCS + pyadjoint + NumPy",
    version=version,
    author="Ivan Yashchuk",
    license="MIT",
    packages=["fecr"],
)
