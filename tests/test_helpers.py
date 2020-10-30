import pytest

import firedrake
import numpy
from firedrake_numpy import firedrake_to_numpy, numpy_to_firedrake


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (firedrake.Constant(0.66), numpy.asarray(0.66)),
        (firedrake.Constant([0.5, 0.66]), numpy.asarray([0.5, 0.66])),
    ],
)
def test_firedrake_to_numpy_constant(test_input, expected):
    assert numpy.allclose(firedrake_to_numpy(test_input), expected)


def test_firedrake_to_numpy_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = firedrake.UnitIntervalMesh(10)
    V = firedrake.FunctionSpace(mesh, "DG", 0)
    x = firedrake.SpatialCoordinate(mesh)
    test_input = firedrake.interpolate(x[0], V)
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(firedrake_to_numpy(test_input), expected)


def test_firedrake_to_numpy_mixed_function():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = firedrake.UnitIntervalMesh(10)
    vec_dim = 4
    V = firedrake.VectorFunctionSpace(mesh, "DG", 0, dim=vec_dim)
    x = firedrake.SpatialCoordinate(mesh)
    test_input = firedrake.interpolate(firedrake.as_vector(vec_dim * (x[0],)), V)
    expected = numpy.linspace(0.05, 0.95, num=10)
    expected = numpy.reshape(numpy.tile(expected, (4, 1)).T, V.dim())
    assert numpy.allclose(firedrake_to_numpy(test_input), expected)


def test_firedrake_to_numpy_vector():
    # Functions in DG0 have nodes at centers of finite element cells
    mesh = firedrake.UnitIntervalMesh(10)
    V = firedrake.FunctionSpace(mesh, "DG", 0)
    x = firedrake.SpatialCoordinate(mesh)
    test_input = firedrake.interpolate(x[0], V)
    test_input_vector = test_input.vector()
    expected = numpy.linspace(0.05, 0.95, num=10)
    assert numpy.allclose(firedrake_to_numpy(test_input_vector), expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (numpy.asarray(0.66), firedrake.Constant(0.66)),
        (numpy.asarray([0.5, 0.66]), firedrake.Constant([0.5, 0.66])),
    ],
)
def test_numpy_to_firedrake_constant(test_input, expected):
    firedrake_test_input = numpy_to_firedrake(test_input, firedrake.Constant(0.0))
    assert numpy.allclose(firedrake_test_input.values(), expected.values())


def test_numpy_to_firedrake_function():
    test_input = numpy.linspace(0.05, 0.95, num=10)
    mesh = firedrake.UnitIntervalMesh(10)
    V = firedrake.FunctionSpace(mesh, "DG", 0)
    template = firedrake.Function(V)
    firedrake_test_input = numpy_to_firedrake(test_input, template)
    x = firedrake.SpatialCoordinate(mesh)
    expected = firedrake.interpolate(x[0], V)
    assert numpy.allclose(
        firedrake_test_input.vector().get_local(), expected.vector().get_local()
    )
