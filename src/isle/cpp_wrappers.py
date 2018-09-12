"""!
Import C++ extension.

Imports everything from isle_cpp into this modules namespace and defines wrappers
around linear algebra types.
"""

import numpy as np

from .isle_cpp import *

class Vector:
    """!
    Wrapper around datatype specific vectors in isle_cpp.

    The constructor returns an instance of isle_cpp.XVector where X is chosen based
    on the argument dtype.
    """

    def __new__(cls, *args, dtype=None, **kwargs):
        """!
        Create and return a Vector for given datatype.
        `dtype` is deduced from other arguments if possible.
        """

        # deduce dtype from numpy array
        if dtype is None and len(args) == 1 and isinstance(args[0], np.ndarray):
            dtype = args[0].dtype

        # include possible numpy types in checks
        if np.issubdtype(dtype, np.floating):
            return DVector(*args, **kwargs)
        if np.issubdtype(dtype, np.signedinteger):
            return IVector(*args, **kwargs)
        if np.issubdtype(dtype, np.complexfloating):
            return CDVector(*args, **kwargs)
        raise ValueError("Datatype not supported: {}".format(dtype))

class Matrix:
    """!
    Wrapper around datatype specific matrices in isle_cpp.

    The constructor returns an instance of isle_cpp.XMatrix where X is chosen based
    on the argument dtype.
    """

    def __new__(cls, *args, dtype=None, **kwargs):
        """!
        Create and return a Matrix for given datatype.
        `dtype` is deduced from other arguments if possible.
        """

        if dtype is None and len(args) == 1:
            # deduce dtype from numpy array
            if isinstance(args[0], np.ndarray):
                dtype = args[0].dtype
            # or deduce from other isle matrix
            elif isinstance(args[0], (DSparseMatrix, DMatrix)):
                dtype = float
            elif isinstance(args[0], (ISparseMatrix, IMatrix)):
                dtype = int
            elif isinstance(args[0], (CDSparseMatrix, CDMatrix)):
                dtype = complex

        # include possible numpy types in checks
        if np.issubdtype(dtype, np.floating):
            return DMatrix(*args, **kwargs)
        if np.issubdtype(dtype, np.signedinteger):
            return IMatrix(*args, **kwargs)
        if np.issubdtype(dtype, np.complexfloating):
            return CDMatrix(*args, **kwargs)
        raise ValueError("Datatype not supported: {}".format(dtype))

class SparseMatrix:
    """!
    Wrapper around datatype specific sparse matrices in isle_cpp.

    The constructor returns an instance of isle_cpp.XSparseMatrix where X is chosen based
    on the argument dtype.
    """

    def __new__(cls, *args, dtype=None, **kwargs):
        """!
        Create and return a SparseMatrix for given datatype.
        `dtype` is deduced from other arguments if possible.
        """

        # deduce dtype from other isle matrix
        if dtype is None and len(args) == 1:
            if isinstance(args[0], (DSparseMatrix, DMatrix)):
                dtype = float
            elif isinstance(args[0], (ISparseMatrix, IMatrix)):
                dtype = int
            elif isinstance(args[0], (CDSparseMatrix, CDMatrix)):
                dtype = complex

        if dtype == float:
            return DSparseMatrix(*args, **kwargs)
        if dtype == int:
            return ISparseMatrix(*args, **kwargs)
        if dtype == complex:
            return CDSparseMatrix(*args, **kwargs)
        raise ValueError("Datatype not supported: {}".format(dtype))
