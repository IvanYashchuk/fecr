# The backends pattern was borrowed from einops package
# https://github.com/arogozhnikov/einops/blob/7a1bfb8738d33bdba3fa94de713814c0b2848c59/einops/_backends.py
"""
Backends in `numpy-fem-adjoint` are organized to meet the following requirements
- backends are not imported unless those are actually needed, because
    - backends may not be installed
    - importing all available backends will drive to significant memory footprint
    - backends may by present but installed with errors (but never used),
      importing may drive to crashes
- backend should be either symbolic or imperative (tensorflow is for both, but that causes problems)
    - this determines which methods (from_numpy/to_numpy or create_symbol/eval_symbol) should be defined
- if backend can't (temporarily) provide symbols for shape dimensions, UnknownSize objects are used
"""

import sys
from typing import Collection
from dataclasses import dataclass
import pyadjoint
import numpy as np

_backends = {}
_debug_importing = False


@dataclass
class BackendVariable:
    block_variable: pyadjoint.block_variable.BlockVariable


class AbstractBackend:
    """Base backend class, major part of methods are only for debugging purposes. """

    framework_name: str

    def is_appropriate_type(self, tensor):
        """ helper method should recognize fem variables it can handle """
        raise NotImplementedError()

    @property
    def Function(self):
        raise NotImplementedError()

    @property
    def lib(self):
        raise NotImplementedError()

    def to_numpy(self, x):
        raise NotImplementedError()

    def from_numpy(self, x, template):
        raise NotImplementedError()

    def __repr__(self):
        return "<numpy-fem-adjoint backend for {}>".format(self.framework_name)


def get_backend(fem_variable) -> AbstractBackend:
    """
    Takes a correct backend (e.g. Firedrake backend if Function is firedrake.Function) for a fem variable.
    If needed, imports package and creates backend
    """
    for framework_name, backend in _backends.items():
        if backend.is_appropriate_type(fem_variable):
            return backend

    # Find backend subclasses recursively
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print("Testing for subclass of ", BackendSubclass)
        if BackendSubclass.framework_name not in _backends:
            # check that module was already imported. Otherwise it can't be imported
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print("Imported backend for ", BackendSubclass.framework_name)
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend
                if backend.is_appropriate_type(fem_variable):
                    return backend

    raise RuntimeError(
        "Type unknown to numpy-fem-adjoint {}".format(type(fem_variable))
    )


class FenicsBackend(AbstractBackend):
    framework_name = "fenics"

    def __init__(self):
        import fenics

        self.fenics = fenics

    @property
    def Function(self):
        return self.fenics.Function

    @property
    def lib(self):
        return self.fenics

    def is_appropriate_type(self, fem_variable):
        if isinstance(fem_variable, self.fenics.Constant):
            return True
        if isinstance(fem_variable, self.fenics.Function):
            return True
        if isinstance(fem_variable, self.fenics.GenericVector):
            return True
        return False

    def to_numpy(self, fenics_var):
        """Convert FEniCS variable to NumPy array.
        Serializes the input so that all MPI ranks have the same data."""
        if isinstance(fenics_var, self.fenics.Constant):
            return np.asarray(fenics_var.values())

        if isinstance(fenics_var, self.fenics.GenericVector):
            if fenics_var.mpi_comm().size > 1:
                data = fenics_var.gather(np.arange(fenics_var.size(), dtype="I"))
            else:
                data = fenics_var.get_local()
            return np.asarray(data)

        if isinstance(fenics_var, self.fenics.Function):
            fenics_vec = fenics_var.vector()
            return self.to_numpy(fenics_vec)

        raise ValueError("Cannot convert " + str(type(fenics_var)))

    def from_numpy(self, numpy_array, fenics_var_template):
        """Convert NumPy array to FEniCS variable.
        Distributes the input array across MPI ranks.
        Input:
            numpy_array (np.array): NumPy array to be converted to FEniCS type
            fenics_var_template (FenicsVariable): Templates for converting arrays to FEniCS type
        Output:
            fenucs_output (FenicsVariable): FEniCS representation of the input numpy_array
        """

        if isinstance(fenics_var_template, self.fenics.Constant):
            if numpy_array.shape == (1,):
                return type(fenics_var_template)(numpy_array[0])
            else:
                return type(fenics_var_template)(numpy_array)

        if isinstance(fenics_var_template, self.fenics.Function):
            function_space = fenics_var_template.function_space()

            u = type(fenics_var_template)(function_space)

            # assume that given numpy array is global array that needs to be distrubuted across processes
            # when FEniCS function is created
            fenics_size = u.vector().size()
            np_size = numpy_array.size

            if np_size != fenics_size:
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
            local_array = numpy_array.reshape(fenics_size)[range_begin:range_end]
            # TODO: replace with a Firedrake-way of setting local portion of data (probably u.dat.data)
            u.vector().set_local(local_array)
            u.vector().apply("insert")
            return u

        err_msg = f"Cannot convert numpy array to {fenics_var_template}"
        raise ValueError(err_msg)


class FiredrakeBackend(AbstractBackend):
    framework_name = "firedrake"

    def __init__(self):
        import firedrake

        self.firedrake = firedrake

    @property
    def Function(self):
        return self.firedrake.Function

    @property
    def lib(self):
        return self.firedrake

    def is_appropriate_type(self, fem_variable):
        if isinstance(fem_variable, self.firedrake.Constant):
            return True
        if isinstance(fem_variable, self.firedrake.Function):
            return True
        if isinstance(fem_variable, self.firedrake.Vector):
            return True
        return False

    def to_numpy(self, firedrake_var):
        """Convert Firedrake variable to NumPy array.
        Serializes the input so that all MPI ranks have the same data."""
        if isinstance(firedrake_var, self.firedrake.Constant):
            return np.asarray(firedrake_var.values())

        if isinstance(firedrake_var, self.firedrake.Function):
            vec = firedrake_var.vector()
            data = vec.gather()
            return np.asarray(data)

        if isinstance(firedrake_var, self.firedrake.Vector):
            data = firedrake_var.gather()
            return np.asarray(data)

        raise ValueError("Cannot convert " + str(type(firedrake_var)))

    def from_numpy(self, numpy_array, firedrake_var_template):
        """Convert numpy array to Firedrake variable.
        Distributes the input array across MPI ranks.
        Input:
            numpy_array (np.array): NumPy array to be converted to Firedrake type
            firedrake_var_template (FiredrakeVariable): Templates for converting arrays to Firedrake type
        Output:
            fenucs_output (FiredrakeVariable): Firedrake representation of the input numpy_array
        """

        if isinstance(firedrake_var_template, self.firedrake.Constant):
            if numpy_array.shape == (1,):
                return type(firedrake_var_template)(numpy_array[0])
            else:
                return type(firedrake_var_template)(numpy_array)

        if isinstance(firedrake_var_template, self.firedrake.Function):
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

        err_msg = f"Cannot convert numpy array to {firedrake_var_template}"
        raise ValueError(err_msg)


class PyadjointBackend(AbstractBackend):
    framework_name = "pyadjoint"

    def __init__(self):
        import pyadjoint

        self.pyadjoint = pyadjoint

    def is_appropriate_type(self, fem_variable):
        return isinstance(fem_variable, (self.pyadjoint.AdjFloat, float))

    def to_numpy(self, pyadjoint_var):
        """Convert pyadjoint float variable to NumPy array."""
        if isinstance(pyadjoint_var, (self.pyadjoint.AdjFloat, float)):
            return np.asarray(pyadjoint_var)
        raise ValueError("Cannot convert " + str(type(pyadjoint_var)))

    def from_numpy(self, numpy_array, pyadjoint_var_template):
        """Convert scalar numpy array to float that pyadjoint understands.
        Input:
            numpy_array (np.array): NumPy array to be converted to pyadjoint type
            pyadjoint_var_template (PyadjointVariable): Templates for converting arrays to pyadjoint type
        Output:
            (FiredrakeVariable): pyadjoint representation of the input numpy_array
        """
        if isinstance(pyadjoint_var_template, self.pyadjoint.AdjFloat):
            return float(numpy_array)
        err_msg = f"Cannot convert numpy array to {pyadjoint_var_template}"
        raise ValueError(err_msg)
