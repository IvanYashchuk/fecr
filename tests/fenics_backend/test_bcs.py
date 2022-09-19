import pytest

import numpy as np

import fenics
import fenics_adjoint as fa
import ufl

import fdm

from fecr import evaluate_primal, evaluate_pullback, evaluate_pushforward
from fecr import to_numpy

mesh = fa.UnitSquareMesh(6, 5)
V = fenics.FunctionSpace(mesh, "P", 1)


def apply_bc(kappa0, kappa1):
    bcs = [
        fa.DirichletBC(V, kappa0, "on_boundary"),
        fa.DirichletBC(V, kappa1, "on_boundary"),
    ]

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )
    u = fa.Function(V)
    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(grad(u), grad(u)) * dx - f * u * dx
    v = fenics.TestFunction(V)
    F = fenics.derivative(JJ, u, v)
    fa.solve(F == 0, u, bcs=bcs)
    return u


templates = (fa.Constant(0.0), fa.Constant(0.0))
inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_fenics_forward():
    numpy_output, _, _, _ = evaluate_primal(apply_bc, templates, *inputs)
    u = apply_bc(fa.Constant(0.5), fa.Constant(0.6))
    assert np.allclose(numpy_output, to_numpy(u))


def test_fenics_vjp():
    numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
        apply_bc, templates, *inputs
    )
    g = np.ones_like(numpy_output)
    vjp_out = evaluate_pullback(fenics_output, fenics_inputs, tape, g)
    check1 = np.isclose(vjp_out[0], np.asarray(42.0))
    check2 = np.isclose(vjp_out[1], np.asarray(42.0))
    assert check1 and check2


def test_fenics_jvp():
    primals = inputs
    tangent0 = np.random.normal(size=(1,))
    tangent1 = np.random.normal(size=(1,))
    tangents = (tangent0, tangent1)

    eval_p = evaluate_primal
    ff0 = lambda x: eval_p(apply_bc, templates, x, primals[1])[0]  # noqa: E731
    ff1 = lambda y: eval_p(apply_bc, templates, primals[0], y)[0]  # noqa: E731
    fdm_jvp0 = fdm.jvp(ff0, tangents[0])(primals[0])
    fdm_jvp1 = fdm.jvp(ff1, tangents[1])(primals[1])

    _, fenics_output, fenics_inputs, tape = evaluate_primal(
        apply_bc, templates, *inputs
    )
    out_tangent = evaluate_pushforward(fenics_output, fenics_inputs, tape, tangents)

    assert np.allclose(fdm_jvp0 + fdm_jvp1, out_tangent)
