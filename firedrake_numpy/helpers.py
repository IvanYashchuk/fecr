import firedrake
import pyadjoint
import numpy as np

import warnings

from typing import Type, List, Union, Iterable, Callable, Tuple

FiredrakeVariable = Union[firedrake.Constant, firedrake.Function, pyadjoint.AdjFloat]


def firedrake_to_numpy(firedrake_var: FiredrakeVariable) -> np.array:
    """Convert Firedrake variable to NumPy array.
    Serializes the input so that all MPI ranks have the same data."""
    if isinstance(firedrake_var, firedrake.Constant):
        return np.asarray(firedrake_var.values())

    if isinstance(firedrake_var, firedrake.Function):
        vec = firedrake_var.vector()
        data = vec.gather()
        return np.asarray(data)

    if isinstance(firedrake_var, firedrake.Vector):
        data = firedrake_var.gather()
        return np.asarray(data)

    if isinstance(firedrake_var, (pyadjoint.AdjFloat, float)):
        return np.asarray(firedrake_var)

    raise ValueError("Cannot convert " + str(type(firedrake_var)))


def numpy_to_firedrake(
    numpy_array: np.array, firedrake_var_template: FiredrakeVariable
) -> FiredrakeVariable:  # noqa: C901
    """Convert numpy array to Firedrake variable.
    Distributes the input array across MPI ranks.
    Input:
        numpy_array (np.array): NumPy array to be converted to Firedrake type
        firedrake_var_template (FiredrakeVariable): Templates for converting arrays to Firedrake type
    Output:
        fenucs_output (FiredrakeVariable): Firedrake representation of the input numpy_array
    """

    if isinstance(firedrake_var_template, firedrake.Constant):
        if numpy_array.shape == (1,):
            return type(firedrake_var_template)(numpy_array[0])
        else:
            return type(firedrake_var_template)(numpy_array)

    if isinstance(firedrake_var_template, firedrake.Function):
        function_space = firedrake_var_template.function_space()

        u = type(firedrake_var_template)(function_space)

        # assume that given numpy array is global array that needs to be distrubuted across processes
        # when Firedrake function is created
        firedrake_size = u.vector().size()
        np_size = numpy_array.size

        if np_size != firedrake_size:
            err_msg = (
                f"Cannot convert numpy array to Function:"
                f"Wrong size {numpy_array.size} vs {u.vector().size()}"
            )
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = (
                f"The numpy array must be of type {np.float_}, "
                "but got {numpy_array.dtype}"
            )
            raise ValueError(err_msg)

        range_begin, range_end = u.vector().local_range()
        numpy_array = np.asarray(numpy_array)
        local_array = numpy_array.reshape(firedrake_size)[range_begin:range_end]
        # TODO: replace with a Firedrake-way of setting local portion of data (probably u.dat.data)
        u.vector().set_local(local_array)
        u.vector().apply("insert")
        return u

    if isinstance(firedrake_var_template, pyadjoint.AdjFloat):
        return float(numpy_array)

    err_msg = f"Cannot convert numpy array to {firedrake_var_template}"
    raise ValueError(err_msg)


def get_numpy_input_templates(
    firedrake_input_templates: Iterable[FiredrakeVariable],
) -> List[np.array]:
    """Returns a list of numpy representations of the input templates"""
    numpy_input_templates = [firedrake_to_numpy(x) for x in firedrake_input_templates]
    return numpy_input_templates


def check_input(
    firedrake_templates: FiredrakeVariable, *args: FiredrakeVariable
) -> None:
    """Checks that the number of inputs arguments is correct"""
    n_args = len(args)
    expected_nargs = len(firedrake_templates)
    if n_args != expected_nargs:
        raise ValueError(
            "Wrong number of arguments"
            " Expected {} got {}.".format(expected_nargs, n_args)
        )

    # Check that each input argument has correct dimensions
    numpy_templates = get_numpy_input_templates(firedrake_templates)
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


def convert_all_to_firedrake(
    firedrake_templates: Iterable[FiredrakeVariable], *args: np.array
) -> List[FiredrakeVariable]:
    """Converts input array to corresponding Firedrake variables"""
    firedrake_inputs = []
    for inp, template in zip(args, firedrake_templates):
        firedrake_inputs.append(numpy_to_firedrake(inp, template))
    return firedrake_inputs
