"""
Basic helpers for test scripts.
"""

from pathlib import Path
import numpy as np

#: Absolute path to test directory.
TEST_PATH = Path(__file__).resolve().parent

#: Precision for test of floating point numbers.
PREC = 1e-15

def prepare_cnxx_import(path=Path("../cnxx/build")):
    """
    Prepare site for scripts based on cnxx. Call before importing cnxx.
    Arguments:
        path: Path to directory that contains cnxx library. Relative to TEST_PATH.
    """

    import site
    site.addsitedir(str(TEST_PATH/path))

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


def equal(a, b):
    "Check whether two numbers or iterables are equal wiht precision core.PREC."

    ahasit = hasattr(a, "__iter__")
    bhasit = hasattr(b, "__iter__")
    if ahasit and bhasit:
        # both are iterables
        if len(a) != len(b):
            return False

        for aelem, belem in zip(a, b):
            if not equal(aelem, belem):
                return False
        return True

    elif not ahasit and not bhasit:
        # both are scalar
        scale = abs(a+b)/2
        comp = abs(a-b)/scale if scale > 1 else abs(a-b)
        if comp < PREC:
            return True
        return False

    else:
        # one is scalar, the other iterable
        return False
