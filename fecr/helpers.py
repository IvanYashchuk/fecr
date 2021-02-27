import numpy as np
import warnings
from typing import Collection
from ._backends import get_backend, BackendVariable


def to_numpy(fem_variable: BackendVariable) -> np.array:
    """Convert fem_variable to NumPy array.
    Serializes the input so that all MPI ranks have the same data."""
    backend = get_backend(fem_variable)
    return backend.to_numpy(fem_variable)


def from_numpy(numpy_array: np.array, backend_var_template: BackendVariable):
    """Convert NumPy array to Firedrake variable.
    Distributes the input array across MPI ranks.
    Input:
        numpy_array (np.array): NumPy array to be converted to FEM backend type
        backend_var_template (BackendVariable): Templates for converting arrays to FEM backend type
    Output:
        (BackendVariable): FEM backend representation of the input numpy_array
    """
    backend = get_backend(backend_var_template)
    return backend.from_numpy(numpy_array, backend_var_template)


def get_numpy_input_templates(
    backend_input_templates: Collection[BackendVariable],
) -> Collection[np.array]:
    """Returns a collection of numpy representations of the input templates"""
    numpy_input_templates = [to_numpy(x) for x in backend_input_templates]
    return numpy_input_templates


def check_input(
    backend_templates: Collection[BackendVariable], *args: np.array
) -> None:
    """Checks that the number of inputs arguments is correct"""
    n_args = len(args)
    expected_nargs = len(backend_templates)
    if n_args != expected_nargs:
        raise ValueError(
            "Wrong number of arguments"
            " Expected {} got {}.".format(expected_nargs, n_args)
        )

    # Check that each input argument has correct dimensions
    numpy_templates = get_numpy_input_templates(backend_templates)
    for i, (arg, template) in enumerate(zip(args, numpy_templates)):
        if arg.shape != template.shape:
            raise ValueError(
                "Expected input shape {} for input"
                " {} but got {}.".format(template.shape, i, arg.shape)
            )

    # Check that the inputs are of double precision
    for i, arg in enumerate(args):
        if arg.dtype != np.float64:
            raise TypeError(
                "All inputs must be type {},"
                " but got {} for input {}.".format(np.float64, arg.dtype, i)
            )


def convert_all_to_backend(
    backend_templates: Collection[BackendVariable], *args: np.array
) -> Collection[BackendVariable]:
    """Converts input array to corresponding backend variables"""
    backend_inputs = []
    for inp, template in zip(args, backend_templates):
        backend_inputs.append(from_numpy(inp, template))
    return backend_inputs
