import firedrake

# import firedrake_adjoint
import pyadjoint

import numpy as np

import functools

from .helpers import (
    numpy_to_firedrake,
    firedrake_to_numpy,
    get_numpy_input_templates,
    check_input,
    convert_all_to_firedrake,
)
from .helpers import FiredrakeVariable

from typing import Type, List, Union, Iterable, Callable, Tuple


def evaluate_primal(
    firedrake_function: Callable,
    firedrake_templates: Iterable[FiredrakeVariable],
    *args: np.array,
) -> Tuple[np.array, FiredrakeVariable, Tuple[FiredrakeVariable], pyadjoint.Tape]:
    """Computes the output of a firedrake_function and saves a corresponding gradient tape
    Input:
        firedrake_function (callable): Firedrake function to be executed during the forward pass
        firedrake_templates (iterable of FiredrakeVariable): Templates for converting arrays to Firedrake types
        args (tuple): NumPy array representation of the input to firedrake_function
    Output:
        numpy_output (np.array): NumPy array representation of the output from firedrake_function(*firedrake_inputs)
        firedrake_output (AdjFloat or Function): Firedrake representation of the output from firedrake_function(*firedrake_inputs)
        firedrake_inputs (list of FiredrakeVariable): Firedrake representation of the input args
        tape (pyadjoint.Tape): pyadjoint's saved computational graph
    """

    check_input(firedrake_templates, *args)
    firedrake_inputs = convert_all_to_firedrake(firedrake_templates, *args)

    # Create tape associated with this forward pass
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    firedrake_output = firedrake_function(*firedrake_inputs)

    if isinstance(firedrake_output, tuple):
        raise ValueError("Only single output from Firedrake function is supported.")

    numpy_output = np.asarray(firedrake_to_numpy(firedrake_output))
    return numpy_output, firedrake_output, firedrake_inputs, tape


# Below unicode symbols are used to distinguish between input and ouput sensitivities
# See http://www.juliadiff.org/ChainRules.jl/dev/FAQ.html
# Δx is the input to a propagator, (i.e a seed for a pullback; or a perturbation for a pushforward)
# ∂x is the output of a propagator
# dx could be either
# Here dx is used for the output of a propagator since ∂x is not a valid name for python variables


def evaluate_vjp(
    dnumpy_output: np.array,
    firedrake_output: FiredrakeVariable,
    firedrake_inputs: Iterable[FiredrakeVariable],
    tape: pyadjoint.Tape,
) -> Tuple[np.array]:
    """Computes the gradients of the output with respect to the inputs.
    Input:
        Δfiredrake_output (np.array): NumPy array representation of the tangent covector to multiply transposed jacobian with
        firedrake_output (AdjFloat or Function): Firedrake representation of the output from firedrake_function(*firedrake_inputs)
        firedrake_inputs (list of FiredrakeVariable): Firedrake representation of the input args
        tape (pyadjoint.Tape): pyadjoint's saved computational graph
    Output:
        dnumpy_inputs (list of np.array):
            NumPy array representation of the `Δfiredrake_output` times jacobian
            of firedrake_function(*firedrake_inputs) wrt to every firedrake_input
    """
    # Convert tangent covector (adjoint variable) to a Firedrake variable
    Δfiredrake_output = numpy_to_firedrake(dnumpy_output, firedrake_output)
    if isinstance(Δfiredrake_output, firedrake.Function):
        Δfiredrake_output = Δfiredrake_output.vector()

    tape.reset_variables()
    firedrake_output.block_variable.adj_value = Δfiredrake_output
    with tape.marked_nodes(firedrake_inputs):
        tape.evaluate_adj(markings=True)
    dfiredrake_inputs = (fi.block_variable.adj_value for fi in firedrake_inputs)

    # Convert Firedrake gradients to NumPy array representation
    dnumpy_inputs = tuple(
        None if dfi is None else np.asarray(firedrake_to_numpy(dfi))
        for dfi in dfiredrake_inputs
    )

    return dnumpy_inputs


def evaluate_jvp(
    firedrake_function: Callable,
    firedrake_templates: Iterable[FiredrakeVariable],
    numpy_inputs: Iterable[np.array],
    Δnumpy_inputs: Iterable[np.array],
) -> Tuple[np.array]:
    """Computes the primal Firedrake function together with the corresponding tangent linear model.
    Note that Δnumpy_inputs are sometimes referred to as tangent vectors.
    """

    numpy_output, firedrake_output, firedrake_inputs, tape = evaluate_primal(
        firedrake_function, firedrake_templates, *numpy_inputs
    )

    # Now tangent (pushforward) evaluation!
    tape.reset_variables()

    Δfiredrake_inputs = convert_all_to_firedrake(firedrake_inputs, *Δnumpy_inputs)
    for fi, Δfi in zip(firedrake_inputs, Δfiredrake_inputs):
        fi.block_variable.tlm_value = Δfi

    tape.evaluate_tlm()

    dfiredrake_output = firedrake_output.block_variable.tlm_value
    dnumpy_output = firedrake_to_numpy(dfiredrake_output)

    return numpy_output, dnumpy_output
