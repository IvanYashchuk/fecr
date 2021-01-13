from pytest_check import check
import numpy as np

import fenics
import fenics_adjoint as fa
import ufl

import fdm

from firedrake_numpy import evaluate_primal, evaluate_vjp, evaluate_jvp


mesh = fa.UnitSquareMesh(3, 2)
V = fenics.FunctionSpace(mesh, "P", 1)


def assemble_fenics(u, kappa0, kappa1):

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    J = fa.assemble(J_form)
    return J


templates = (fa.Function(V), fa.Constant(0.0), fa.Constant(0.0))
inputs = (np.ones(V.dim()), np.ones(1) * 0.5, np.ones(1) * 0.6)
ff = lambda *args: evaluate_primal(assemble_fenics, templates, *args)[0]  # noqa: E731
ff0 = lambda x: ff(x, inputs[1], inputs[2])  # noqa: E731
ff1 = lambda y: ff(inputs[0], y, inputs[2])  # noqa: E731
ff2 = lambda z: ff(inputs[0], inputs[1], z)  # noqa: E731


def test_fenics_forward():
    numpy_output, _, _, _, = evaluate_primal(assemble_fenics, templates, *inputs)
    u1 = fa.interpolate(fa.Constant(1.0), V)
    J = assemble_fenics(u1, fa.Constant(0.5), fa.Constant(0.6))
    assert np.isclose(numpy_output, J)


def test_vjp_assemble_eval():
    numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
        assemble_fenics, templates, *inputs
    )
    g = np.ones_like(numpy_output)
    vjp_out = evaluate_vjp(g, fenics_output, fenics_inputs, tape)

    fdm_jac0 = fdm.jacobian(ff0)(inputs[0])
    fdm_jac1 = fdm.jacobian(ff1)(inputs[1])
    fdm_jac2 = fdm.jacobian(ff2)(inputs[2])

    check1 = np.allclose(vjp_out[0], fdm_jac0)
    check2 = np.allclose(vjp_out[1], fdm_jac1)
    check3 = np.allclose(vjp_out[2], fdm_jac2)
    assert check1 and check2 and check3


def test_jvp_assemble_eval():
    primals = inputs
    tangent0 = np.random.normal(size=(V.dim(),))
    tangent1 = np.random.normal(size=(1,))
    tangent2 = np.random.normal(size=(1,))
    tangents = (tangent0, tangent1, tangent2)

    fdm_jvp0 = fdm.jvp(ff0, tangents[0])(primals[0])
    fdm_jvp1 = fdm.jvp(ff1, tangents[1])(primals[1])
    fdm_jvp2 = fdm.jvp(ff2, tangents[2])(primals[2])

    _, out_tangent = evaluate_jvp(assemble_fenics, templates, primals, tangents)

    assert np.allclose(fdm_jvp0 + fdm_jvp1 + fdm_jvp2, out_tangent)
