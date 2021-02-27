import pytest

import numpy as np

import firedrake
import firedrake_adjoint
import ufl

import fdm

from fecr import evaluate_primal, evaluate_pullback, evaluate_pushforward
from fecr import to_numpy

mesh = firedrake.UnitSquareMesh(6, 5)
V = firedrake.FunctionSpace(mesh, "P", 1)


def solve_firedrake(kappa0, kappa1):

    x = firedrake.SpatialCoordinate(mesh)
    f = x[0]

    u = firedrake.Function(V)
    bcs = [firedrake.DirichletBC(V, firedrake.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = firedrake.TestFunction(V)
    F = firedrake.derivative(JJ, u, v)
    firedrake.solve(F == 0, u, bcs=bcs)
    return u


templates = (firedrake.Constant(0.0), firedrake.Constant(0.0))
inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_firedrake_forward():
    numpy_output, _, _, _ = evaluate_primal(solve_firedrake, templates, *inputs)
    u = solve_firedrake(firedrake.Constant(0.5), firedrake.Constant(0.6))
    assert np.allclose(numpy_output, to_numpy(u))


def test_firedrake_vjp():
    numpy_output, firedrake_output, firedrake_inputs, tape = evaluate_primal(
        solve_firedrake, templates, *inputs
    )
    g = np.ones_like(numpy_output)
    vjp_out = evaluate_pullback(firedrake_output, firedrake_inputs, tape, g)
    check1 = np.isclose(vjp_out[0], np.asarray(-1.13533304))
    check2 = np.isclose(vjp_out[1], np.asarray(0.94611087))
    assert check1 and check2


def test_firedrake_jvp():
    primals = inputs
    tangent0 = np.random.normal(size=(1,))
    tangent1 = np.random.normal(size=(1,))
    tangents = (tangent0, tangent1)

    eval_p = evaluate_primal
    ff0 = lambda x: eval_p(solve_firedrake, templates, x, primals[1])[0]  # noqa: E731
    ff1 = lambda y: eval_p(solve_firedrake, templates, primals[0], y)[0]  # noqa: E731
    fdm_jvp0 = fdm.jvp(ff0, tangents[0])(primals[0])
    fdm_jvp1 = fdm.jvp(ff1, tangents[1])(primals[1])

    _, firedrake_output, firedrake_inputs, tape = evaluate_primal(
        solve_firedrake, templates, *inputs
    )
    out_tangent = evaluate_pushforward(
        firedrake_output, firedrake_inputs, tape, tangents
    )

    assert np.allclose(fdm_jvp0 + fdm_jvp1, out_tangent)
