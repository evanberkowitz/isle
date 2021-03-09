"""
Basic helpers for test scripts.
"""

from pathlib import Path
import numpy as np
import operator

#: Absolute path to test directory.
TEST_PATH = Path(__file__).resolve().parent

def get_logger():
    """
    Get a logger to report messages.
    Can be called multiple times; always returns the same instance.
    """

    if not get_logger.logger:
        import logging

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s',
                                          "%Y-%m-%d %H:%M"))

        get_logger.logger = logging.getLogger()
        get_logger.logger.setLevel(logging.INFO)
        get_logger.logger.addHandler(ch)

    return get_logger.logger
get_logger.logger = None


def type_eq(typ0, typ1):
    "Check if two types are equal, taking special numpy types into account."
    if typ0 == np.dtype("int64"):
        typ0 = int
    if typ1 == np.dtype("int64"):
        typ1 = int
    return typ0 == typ1


def isEqual(a, b, nOps=1, prec=1e-14):
    """
    Check whether two numbers or iterables are equal with precision core.PREC.
    For large scale operations, multiply scale by nOps for the check.
    """

    ahasit = hasattr(a, "__iter__")
    bhasit = hasattr(b, "__iter__")
    if ahasit and bhasit:
        # both are iterables
        if len(a) != len(b):
            return False

        for aelem, belem in zip(a, b):
            if not isEqual(aelem, belem, nOps=nOps, prec=prec):
                return False
        return True

    elif not ahasit and not bhasit:
        # both are scalar
        scale = abs(a+b)/2
        comp = abs(a-b)/scale if scale > 1 else abs(a-b)

        if comp < prec*nOps:
            return True
        return False

    else:
        # one is scalar, the other iterable
        return False


OperatorDict = {
  '+'  : operator.add,
  '+=' : operator.iadd,
  '-'  : operator.sub,
  '-=' : operator.isub,
  '*'  : operator.mul,
  '*=' : operator.imul,
  '/'  : operator.truediv,
  '//' : operator.floordiv,
  '/=' : operator.itruediv,
  '@'  : operator.matmul
}
