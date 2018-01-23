"""
Imports everything from cnxx into this modules namespace and defines wrappers
around linear algebra types.
"""

from cns.core import prepare_cnxx_import

prepare_cnxx_import()
from cnxx import *


class Vector:
    """
    Wrapper around datatype specific vectors in cnxx.

    The constructor returns an instance of cnxx.XVector where X is chosen based
    on the argument dtype.
    """

    def __new__(cls, *args, dtype=float, **kwargs):
        "Create and return a Vector for given datatype."
        if dtype == float:
            return DVector(*args, **kwargs)
        if dtype == int:
            return IVector(*args, **kwargs)
        if dtype == complex:
            return CDVector(*args, **kwargs)
        raise ValueError("Datatype not supported: {}".format(dtype))

class Matrix:
    """
    Wrapper around datatype specific matrices in cnxx.

    The constructor returns an instance of cnxx.XMatrix where X is chosen based
    on the argument dtype.
    """

    def __new__(cls, *args, dtype=float, **kwargs):
        "Create and return a Matrix for given datatype."
        if dtype == float:
            return DMatrix(*args, **kwargs)
        if dtype == int:
            return IMatrix(*args, **kwargs)
        if dtype == complex:
            return CDMatrix(*args, **kwargs)
        raise ValueError("Datatype not supported: {}".format(dtype))

class SparseMatrix:
    """
    Wrapper around datatype specific sparse matrices in cnxx.

    The constructor returns an instance of cnxx.XSparseMatrix where X is chosen based
    on the argument dtype.
    """

    def __new__(cls, *args, dtype=float, **kwargs):
        "Create and return a SparseMatrix for given datatype."
        if dtype == float:
            return DSparseMatrix(*args, **kwargs)
        if dtype == int:
            return ISparseMatrix(*args, **kwargs)
        if dtype == complex:
            return CDSparseMatrix(*args, **kwargs)
        raise ValueError("Datatype not supported: {}".format(dtype))
