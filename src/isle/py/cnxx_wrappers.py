"""!
Import cnxx.

Imports everything from cnxx into this modules namespace and defines wrappers
around linear algebra types.
"""

import numpy as np

from cns.core import prepare_cnxx_import
prepare_cnxx_import()
from cnxx import *


class Vector:
    """!
    Wrapper around datatype specific vectors in cnxx.

    The constructor returns an instance of cnxx.XVector where X is chosen based
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
    Wrapper around datatype specific matrices in cnxx.

    The constructor returns an instance of cnxx.XMatrix where X is chosen based
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
            # or deduce from other cns matrix
            elif isinstance(args[0], DSparseMatrix) or isinstance(args[0], DMatrix):
                dtype = float
            elif isinstance(args[0], ISparseMatrix) or isinstance(args[0], IMatrix):
                dtype = int
            elif isinstance(args[0], CDSparseMatrix) or isinstance(args[0], CDMatrix):
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
    Wrapper around datatype specific sparse matrices in cnxx.

    The constructor returns an instance of cnxx.XSparseMatrix where X is chosen based
    on the argument dtype.
    """

    def __new__(cls, *args, dtype=None, **kwargs):
        """!
        Create and return a SparseMatrix for given datatype.
        `dtype` is deduced from other arguments if possible.
        """

        # deduce dtype from other cns matrix
        if dtype is None and len(args) == 1:
            if isinstance(args[0], DSparseMatrix) or isinstance(args[0], DMatrix):
                dtype = float
            elif isinstance(args[0], ISparseMatrix) or isinstance(args[0], IMatrix):
                dtype = int
            elif isinstance(args[0], CDSparseMatrix) or isinstance(args[0], CDMatrix):
                dtype = complex

        if dtype == float:
            return DSparseMatrix(*args, **kwargs)
        if dtype == int:
            return ISparseMatrix(*args, **kwargs)
        if dtype == complex:
            return CDSparseMatrix(*args, **kwargs)
        raise ValueError("Datatype not supported: {}".format(dtype))
