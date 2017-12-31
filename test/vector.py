#!/usr/bin/env python

"""
Example python module for unittests.
"""

import unittest
import operator
import numpy as np

import core
core.prepare_cnxx_import()
import cnxx
import rand

# RNG params
SEED = 1
RAND_MIN = -1
RAND_MAX = +1


def _vec_div(num, denom):
    "Divide two vector, where zeros in denom are set to 1."
    denom = type(denom)([d if d != 0 else 1 for d in denom])
    return num/denom

def _arr_div(num, denom):
    "Divide two arrays, where zeros in denom are set to 1."
    denom = np.array([d if d != 0 else 1 for d in denom], dtype=denom.dtype)
    op = operator.floordiv if num.dtype == np.dtype("int64") else operator.truediv
    return op(num, denom)


class TestVector(unittest.TestCase):
    N = 100
    TYPES = (int, float, complex)
    VTYPE_PER_TYPE = {int: cnxx.IVector,
                      float: cnxx.DVector,
                      complex: cnxx.CDVector}

    def _test_size_construction(self, vtyp, typ):
        vec = vtyp(1)
        self.assertIsInstance(vec[0], typ,
                              "Type check for construction of vector with given size")

    def _test_buffer_construction(self, vtyp, typ):
        array = rand.randn(RAND_MIN, RAND_MAX, self.N, typ)
        vec = vtyp(array)
        self.assertIsInstance(vec[0], typ,
                              "Type check for construction of vector via buffer protocol")
        for velem, lelem in zip(vec, array):
            self.assertEqual(velem, lelem,
                             "Value check for construction of vector via buffer protocol")

    def _test_list_construction(self, vtyp, typ):
        array = rand.randn(RAND_MIN, RAND_MAX, self.N, typ)
        vec = vtyp(list(array))
        self.assertIsInstance(vec[0], typ,
                              "Type check for construction of vector from list")
        for velem, lelem in zip(vec, array):
            self.assertEqual(velem, lelem,
                             "Value check for construction of vector from list")

    def test_construction(self):
        for typ in self.TYPES:
            vtyp = self.VTYPE_PER_TYPE[typ]
            self._test_size_construction(vtyp, typ)
            # self._test_buffer_construction(vtyp, typ)
            self._test_list_construction(vtyp, typ)


    def _test_op_v(self, res, npres, opname):
        "Test an operation with a vector result."
        self.assertTrue(core.type_eq(type(res[0]), npres.dtype),
                        "Type check for operation {}".format(opname))
        self.assertTrue(core.equal(res, npres),
                        "Value check for operation {}".format(opname))

    def _test_op_s(self, res, npres, opname):
        "Test an operation with a scalar result."
        self.assertIsInstance(res, type(npres),
                              "Type check for operation {}".format(opname))
        self.assertTrue(core.equal(res, npres),
                        "Value check for operation {}".format(opname))

    def test_operators(self):
        iarray0 = rand.randn(RAND_MIN, RAND_MAX, self.N, int)
        iarray1 = rand.randn(RAND_MIN, RAND_MAX, self.N, int)
        darray0 = rand.randn(RAND_MIN, RAND_MAX, self.N, float)
        darray1 = rand.randn(RAND_MIN, RAND_MAX, self.N, float)
        cdarray0 = rand.randn(RAND_MIN, RAND_MAX, self.N, complex)
        cdarray1 = rand.randn(RAND_MIN, RAND_MAX, self.N, complex)

        # TODO use buffer construction when fixed
        ivec0 = cnxx.IVector(list(iarray0))
        ivec1 = cnxx.IVector(list(iarray1))
        dvec0 = cnxx.DVector(darray0)
        dvec1 = cnxx.DVector(darray1)
        cdvec0 = cnxx.CDVector(cdarray0)
        cdvec1 = cnxx.CDVector(cdarray1)

        # two vectors
        for vec0, vec1, arr0, arr1 in ((ivec0, ivec1, iarray0, iarray1),
                                       (dvec0, dvec1, darray0, darray1),
                                       (cdvec0, cdvec1, cdarray0, cdarray1)):
            self._test_op_v(vec0+vec1, arr0+arr1, "vecadd")
            self._test_op_v(vec0-vec1, arr0-arr1, "vecsub")
            self._test_op_v(vec0*vec1, arr0*arr1, "vecmul")
            self._test_op_v(_vec_div(vec0, vec1), _arr_div(arr0, arr1), "vecdiv")


def setUpModule():
    "Setup the vector test module."

    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    min:  {}
    max:  {}""".format(SEED, RAND_MIN, RAND_MAX))

    rand.setup(SEED)
