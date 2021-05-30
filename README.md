# Finite Element Chain Rules &middot; [![Build FEniCS](https://github.com/ivanyashchuk/fecr/workflows/FEniCS/badge.svg)](https://github.com/ivanyashchuk/fecr/actions?query=workflow%3AFEniCS+branch%3Amaster) [![Build Firedrake](https://github.com/ivanyashchuk/fecr/workflows/Firedrake/badge.svg)](https://github.com/ivanyashchuk/fecr/actions?query=workflow%3AFiredrake+branch%3Amaster) [![codecov](https://codecov.io/gh/IvanYashchuk/fecr/branch/master/graph/badge.svg?token=5TCRI7OT6E)](https://codecov.io/gh/IvanYashchuk/fecr)

Easy interoperability with Automatic Differentiation libraries through NumPy interface to Firedrake and FEniCS.

## Overview
This package provides a high-level interface for evaluating derivatives of [FEniCS](http://fenicsproject.org) and [Firedrake](http://firedrakeproject.org/) models.
It is intended to be used as the backend for extending Automatic Differentiation (AD) libraries to support FEniCS and Firedrake solvers.

Automatic tangent linear and adjoint solvers for FEniCS/Firedrake problems are derived with [dolfin-adjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/).
These solvers make it possible to use forward and reverse modes Automatic Differentiation with FEniCS/Firedrake.

This package is used for building bridges between FEniCS/Firedrake and:
* JAX in [jax-fenics-adjoint](https://github.com/IvanYashchuk/jax-fenics-adjoint/),
* PyMC3 (Theano) in [fenics-pymc3](https://github.com/IvanYashchuk/fenics-pymc3),
* Julia's ChainRules.jl in [PyFenicsAD.jl](https://github.com/IvanYashchuk/PyFenicsAD.jl).

Current limitations:
* Composition of forward and reverse modes for higher-order derivatives is not implemented yet.
* Differentiation wrt Dirichlet boundary conditions and mesh coordinates is not implemented yet.

## API
The package includes 5 functions:

 - two functions for converting between NumPy and FEniCS/Firedrake: `to_numpy`, `from_numpy`,
 - `evaluate_primal` - computes the output of a FEniCS/Firedrake function and saves a corresponding computational graph,
 - `evaluate_pullback` - propagates the derivative information from outputs to inputs (reverse-mode AD),
 - `evaluate_pushforward` - propagates the derivative information from inputs to outputs (forward-mode AD).

## Example
Here is the demonstration of solving the [Poisson's PDE](https://en.wikipedia.org/wiki/Poisson%27s_equation)
on the 2D square domain and calculating the result of multiplying a vector with the solution Jacobian matrix (_du/df_) using the reverse mode Automatic Differentiation.
```python
import numpy as np

import firedrake
import firedrake_adjoint
import ufl

from functools import partial

from fecr import evaluate_primal, evaluate_pullback
from fecr import from_numpy

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
vjp_out = evaluate_pullback(firedrake_output, firedrake_inputs, tape, g)
```

Check the `tests/` folder for the additional usage examples.

## Installation
First install [FEniCS](https://fenicsproject.org/download/) or [Firedrake](https://firedrakeproject.org/download.html).
Then install [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) with:

    python -m pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@master

Then install [firedrake-numpy-adjoint](https://github.com/IvanYashchuk/firedrake-numpy-adjoint) with:

    python -m pip install git+https://github.com/IvanYashchuk/firedrake-numpy-adjoint@master


## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/firedrake-numpy-adjoint/issues/new

## Asking questions and general discussion

If you have a question or anything else, create a new [discussion]. Using issues is also fine!

[discussion]: https://github.com/IvanYashchuk/firedrake-numpy-adjoint/discussions/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/firedrake-numpy-adjoint.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/firedrake_backend  # or pytest tests/fenics_backend

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/firedrake-numpy-adjoint/pulls
