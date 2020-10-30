import sys

from setuptools import setup

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

version = "0.1"

setup(
    name="firedrake_numpy",
    description="Firedrake adjoint + NumPy",
    version=version,
    author="Ivan Yashchuk",
    license="MIT",
    packages=["firedrake_numpy"],
)
