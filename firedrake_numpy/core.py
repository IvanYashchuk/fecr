import pyadjoint
import numpy as np
import functools

from .helpers import (
    from_numpy,
    to_numpy,
    get_numpy_input_templates,
    check_input,
    convert_all_to_backend,
)
from ._backends import BackendVariable, get_backend

from typing import Type, Collection, Callable, Tuple


def evaluate_primal(
    firedrake_function: Callable[..., BackendVariable],
    firedrake_templates: Collection[BackendVariable],
    *args: np.array,
) -> Tuple[np.array, BackendVariable, Collection[BackendVariable], pyadjoint.Tape]:
    """Computes the output of a firedrake_function and saves a corresponding gradient tape
    Input:
        firedrake_function (callable): Firedrake function to be executed during the forward pass
        firedrake_templates (collection of BackendVariable): Templates for converting arrays to Firedrake types
        args (tuple): NumPy array representation of the input to firedrake_function
    Output:
        numpy_output (np.array): NumPy array representation of the output from firedrake_function(*firedrake_inputs)
        firedrake_output (AdjFloat or Function): Firedrake representation of the output from firedrake_function(*firedrake_inputs)
        firedrake_inputs (collection of BackendVariable): Firedrake representation of the input args
        tape (pyadjoint.Tape): pyadjoint's saved computational graph
    """

    check_input(firedrake_templates, *args)
    firedrake_inputs = convert_all_to_backend(firedrake_templates, *args)

    # Create tape associated with this forward pass
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    firedrake_output = firedrake_function(*firedrake_inputs)

    if isinstance(firedrake_output, tuple):
        raise ValueError("Only single output from Firedrake function is supported.")

    numpy_output = np.asarray(to_numpy(firedrake_output))
    return numpy_output, firedrake_output, firedrake_inputs, tape


# Below unicode symbols are used to distinguish between input and ouput sensitivities
# See http://www.juliadiff.org/ChainRules.jl/dev/FAQ.html
# Δx is the input to a propagator, (i.e a seed for a pullback; or a perturbation for a pushforward)
# ∂x is the output of a propagator
# dx could be either
# Here dx is used for the output of a propagator since ∂x is not a valid name for python variables


def evaluate_pullback(
    firedrake_output: BackendVariable,
    firedrake_inputs: Collection[BackendVariable],
    tape: pyadjoint.Tape,
    Δnumpy_output: np.array,
) -> Collection[np.array]:
    """Pullback is a function to propagate the derivative information from outputs to inputs.
    It also corresponds to evaluating a Jacobian transpose vector product or vector Jacobian product.
    This is a reverse-mode automatic differentiation.
    Input:
        firedrake_output (AdjFloat or Function): Firedrake representation of the output from firedrake_function(*firedrake_inputs)
        firedrake_inputs (collection of BackendVariable): Firedrake representation of the input args
        tape (pyadjoint.Tape): pyadjoint's saved computational graph
        Δnumpy_output (np.array): NumPy array representation of the tangent covector to multiply transposed Jacobian with
    Output:
        dnumpy_inputs (collection of np.array):
            NumPy array representation of the `Δnumpy_output` times Jacobian
            of firedrake_function(*firedrake_inputs) wrt to every firedrake_input
    """
    # Convert tangent covector (adjoint variable) to a backend variable
    Δfiredrake_output = from_numpy(Δnumpy_output, firedrake_output)

    # pyadjoint doesn't allow setting Functions to block_variable.adj_value
    backend = get_backend(firedrake_inputs[0])
    if isinstance(Δfiredrake_output, backend.Function):
        Δfiredrake_output = Δfiredrake_output.vector()

    tape.reset_variables()
    firedrake_output.block_variable.adj_value = Δfiredrake_output
    with tape.marked_nodes(firedrake_inputs):
        tape.evaluate_adj(markings=True)
    dfiredrake_inputs = (fi.block_variable.adj_value for fi in firedrake_inputs)

    # Convert Firedrake gradients to NumPy array representation
    dnumpy_inputs = tuple(
        None if dfi is None else np.asarray(to_numpy(dfi)) for dfi in dfiredrake_inputs
    )

    return dnumpy_inputs


def evaluate_pushforward(
    firedrake_output: BackendVariable,
    firedrake_inputs: Collection[BackendVariable],
    tape: pyadjoint.Tape,
    Δnumpy_inputs: Collection[np.array],
) -> Collection[np.array]:
    """Pushforward is a function to propagate the derivative information from inputs to outputs.
    It also corresponds to evaluating a Jacobian vector product.
    This is a forward-mode automatic differentiation.
    Input:
        firedrake_output (AdjFloat or Function): Firedrake representation of the output from firedrake_function(*firedrake_inputs)
        firedrake_inputs (collection of BackendVariable): Firedrake representation of the input args
        tape (pyadjoint.Tape): pyadjoint's saved computational graph
        Δnumpy_inputs (collection of np.array): NumPy array representation of the tangent vector to multiply with Jacobian
    Output:
        dnumpy_output (np.array):
            NumPy array representation of the `Δnumpy_inputs` times Jacobian
            of firedrake_function(*firedrake_inputs) wrt to every firedrake_input
    """

    # Now tangent (pushforward) evaluation!
    tape.reset_variables()

    Δfiredrake_inputs = convert_all_to_backend(firedrake_inputs, *Δnumpy_inputs)
    for fi, Δfi in zip(firedrake_inputs, Δfiredrake_inputs):
        fi.block_variable.tlm_value = Δfi

    tape.evaluate_tlm()

    dfiredrake_output = firedrake_output.block_variable.tlm_value
    dnumpy_output = to_numpy(dfiredrake_output)

    return dnumpy_output
