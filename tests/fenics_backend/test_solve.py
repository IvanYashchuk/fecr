import pytest

import numpy as np

import fenics
import fenics_adjoint as fa
import ufl

import fdm

from fenics_numpy import evaluate_primal, evaluate_vjp, evaluate_jvp
from fenics_numpy import to_numpy

mesh = fa.UnitSquareMesh(6, 5)
V = fenics.FunctionSpace(mesh, "P", 1)


def solve_fenics(kappa0, kappa1):

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    u = fa.Function(V)
    bcs = [fa.DirichletBC(V, fa.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = fenics.TestFunction(V)
    F = fenics.derivative(JJ, u, v)
    fa.solve(F == 0, u, bcs=bcs)
    return u


templates = (fa.Constant(0.0), fa.Constant(0.0))
inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_fenics_forward():
    numpy_output, _, _, _ = evaluate_primal(solve_fenics, templates, *inputs)
    u = solve_fenics(fa.Constant(0.5), fa.Constant(0.6))
    assert np.allclose(numpy_output, to_numpy(u))


def test_fenics_vjp():
    numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
        solve_fenics, templates, *inputs
    )
    g = np.ones_like(numpy_output)
    vjp_out = evaluate_vjp(g, fenics_output, fenics_inputs, tape)
    check1 = np.isclose(vjp_out[0], np.asarray(-2.91792642))
    check2 = np.isclose(vjp_out[1], np.asarray(2.43160535))
    assert check1 and check2


def test_fenics_jvp():
    primals = inputs
    tangent0 = np.random.normal(size=(1,))
    tangent1 = np.random.normal(size=(1,))
    tangents = (tangent0, tangent1)

    eval_p = evaluate_primal
    ff0 = lambda x: eval_p(solve_fenics, templates, x, primals[1])[0]  # noqa: E731
    ff1 = lambda y: eval_p(solve_fenics, templates, primals[0], y)[0]  # noqa: E731
    fdm_jvp0 = fdm.jvp(ff0, tangents[0])(primals[0])
    fdm_jvp1 = fdm.jvp(ff1, tangents[1])(primals[1])

    _, out_tangent = evaluate_jvp(solve_fenics, templates, primals, tangents)

    assert np.allclose(fdm_jvp0 + fdm_jvp1, out_tangent)
