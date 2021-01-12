# firedrake-numpy-adjoint &middot; [![Build](https://github.com/ivanyashchuk/firedrake-numpy-adjoint/workflows/CI/badge.svg)](https://github.com/ivanyashchuk/firedrake-numpy-adjoint/actions?query=workflow%3ACI+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/IvanYashchuk/firedrake-numpy-adjoint/badge.svg?branch=master)](https://coveralls.io/github/IvanYashchuk/firedrake-numpy-adjoint?branch=master)

Easy interoperability with Automatic Differentiation libraries through NumPy interface to Firedrake.

## Example
Here is the demonstration of solving the [Poisson's PDE](https://en.wikipedia.org/wiki/Poisson%27s_equation)
on the 2D square domain and calculating the result of multiplying a vector with the solution Jacobian matrix (_du/df_) using the reverse (adjoint) mode Automatic Differentiation.
```python
import numpy as np

import firedrake
import firedrake_adjoint
import ufl

from functools import partial

from firedrake_numpy import evaluate_primal, evaluate_vjp
from firedrake_numpy import from_numpy

# Create mesh for the unit square domain
n = 10
mesh = firedrake.UnitSquareMesh(n, n)

# Define discrete function spaces and functions
V = firedrake.FunctionSpace(mesh, "CG", 1)
W = firedrake.FunctionSpace(mesh, "DG", 0)

# Define Firedrake template representation of NumPy input
templates = (firedrake.Function(W),)

# This function takes Firedrake types as arguments and returns a Firedrake Function (solution)
def firedrake_solve(f):
    # This function inside should be traceable by firedrake_adjoint
    u = firedrake.Function(V, name="PDE Solution")
    v = firedrake.TestFunction(V)
    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    F = (inner(grad(u), grad(v)) - f * v) * dx
    bcs = [firedrake.DirichletBC(V, 0.0, "on_boundary")]
    firedrake.solve(F == 0, u, bcs)
    return u

# Let's build a decorator which transforms NumPy input to Firedrake types input
# and returns NumPy representation of Firedrake output
numpy_firedrake_solve = partial(evaluate_primal, firedrake_solve, templates)

# Let's create a vector of ones with size equal to the number of cells in the mesh
f = np.ones(W.dim())
u = numpy_firedrake_solve(f)[0] # u is a NumPy array now
u_firedrake = from_numpy(u, firedrake.Function(V)) # we need to explicitly provide template function for conversion

# Now let's evaluate the vector-Jacobian product
numpy_output, firedrake_output, firedrake_inputs, tape = numpy_firedrake_solve(f)
g = np.ones_like(numpy_output)

# `vjp_out` is the result of (implicitly) multiplying the vector `g` with the solution Jacobian du/df
vjp_out = evaluate_vjp(g, firedrake_output, firedrake_inputs, tape)
```

Check the `tests/` folder for the additional usage examples.
