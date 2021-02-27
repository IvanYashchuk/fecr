import pytest
from os.path import abspath, basename, dirname, join
import subprocess
import glob
import sys


cwd = abspath(dirname(__file__))
fenics_tests_dir = join(cwd, "fenics_backend")
firedrake_tests_dir = join(cwd, "firedrake_backend")


@pytest.fixture(params=glob.glob(f"{fenics_tests_dir}/*.py"), ids=lambda x: basename(x))
def fenics_py_file(request):
    return abspath(request.param)


@pytest.fixture(
    params=glob.glob(f"{firedrake_tests_dir}/*.py"), ids=lambda x: basename(x)
)
def firedrake_py_file(request):
    return abspath(request.param)


@pytest.mark.fenics
def test_fenics_runs(fenics_py_file):
    subprocess.check_call([sys.executable, fenics_py_file])


@pytest.mark.firedrake
def test_firedrake_runs(firedrake_py_file):
    subprocess.check_call([sys.executable, firedrake_py_file])
