import pytest

import firedrake
import numpy
from fecr import to_numpy, from_numpy


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (firedrake.Constant(0.66), numpy.asarray(0.66)),
        (firedrake.Constant([0.5, 0.66]), numpy.asarray([0.5, 0.66])),
    ],
)
def test_firedrake_to_numpy_constant(test_input, expected):
    assert numpy.allclose(to_numpy(test_input), expected)


def test_firedrake_to_numpy_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = firedrake.UnitIntervalMesh(10)
    V = firedrake.FunctionSpace(mesh, "DG", 0)
    x = firedrake.SpatialCoordinate(mesh)
    test_input = firedrake.interpolate(x[0], V)
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(to_numpy(test_input), expected)


def test_firedrake_to_numpy_mixed_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = firedrake.UnitIntervalMesh(10)
    vec_dim = 4
    V = firedrake.VectorFunctionSpace(mesh, "DG", 0, dim=vec_dim)
    x = firedrake.SpatialCoordinate(mesh)
    test_input = firedrake.interpolate(firedrake.as_vector(vec_dim * (x[0],)), V)
    expected = numpy.linspace(0.05, 0.95, num=10)
    expected = numpy.reshape(numpy.tile(expected, (4, 1)).T, V.dim())
    assert numpy.allclose(to_numpy(test_input), expected)


def test_firedrake_to_numpy_vector():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = firedrake.UnitIntervalMesh(10)
    V = firedrake.FunctionSpace(mesh, "DG", 0)
    x = firedrake.SpatialCoordinate(mesh)
    test_input = firedrake.interpolate(x[0], V)
    test_input_vector = test_input.vector()
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(to_numpy(test_input_vector), expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (numpy.asarray(0.66), firedrake.Constant(0.66)),
        (numpy.asarray([0.5, 0.66]), firedrake.Constant([0.5, 0.66])),
    ],
)
def test_numpy_to_firedrake_constant(test_input, expected):
    firedrake_test_input = from_numpy(test_input, firedrake.Constant(0.0))
    assert numpy.allclose(firedrake_test_input.values(), expected.values())


@pytest.mark.parametrize(
    "dtype",
    [
        numpy.float64,
        # Default build of PETSc supports only float64 dtype
        # AssertionError: Can't create Vec with type float32, must be <class 'numpy.float64'>
        # PyOP2/pyop2/types/dat.py:683: AssertionError
        pytest.param(numpy.float32, marks=pytest.mark.xfail),
        pytest.param(numpy.complex64, marks=pytest.mark.xfail),
        pytest.param(numpy.complex128, marks=pytest.mark.xfail),
    ],
)
def test_numpy_to_firedrake_function(dtype):
    test_input = numpy.linspace(0.05, 0.95, num=10, dtype=dtype)
    mesh = firedrake.UnitIntervalMesh(10)
    V = firedrake.FunctionSpace(mesh, "DG", 0)
    template = firedrake.Function(V, dtype=dtype)
    firedrake_test_input = from_numpy(test_input, template)
    x = firedrake.SpatialCoordinate(mesh)
    expected = firedrake.interpolate(x[0], V)
    assert numpy.allclose(
        firedrake_test_input.vector().get_local(), expected.vector().get_local()
    )
