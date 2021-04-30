#!/usr/bin/env python

"""
Unittest for wrappers around blaze vector classes.
"""

import unittest
import abc
import numpy as np

import isle
from . import rand
from . import core

# RNG params
SEED = 1
RAND_MIN = -5
RAND_MAX = +5


class AbstractVectorTest(metaclass=abc.ABCMeta):
    "Abstract vector class test. Vector test must inherit from this class"
    Nvec        = None  # Size of tested vectors
    cVecTypes   = []    # Vector types for cVecs
    scalarTypes = {}    # Vector types for given scalars
    operations  = {}    # Dict for matrix vector multiplications

#-------------- Constructors --------------
    def _test_size_construction(self, vtyp):
        "Test construction by size and type checks vector with given size"
        typ = self.scalarTypes[vtyp]
        cVec = vtyp(1)
        self.assertIsInstance(
            cVec[0],
            typ,
            msg="Failed type check for scalar type: {typ} and vector type: {vtyp}".format(
              typ=typ, vtyp=vtyp
            )
        )

    #----------------------------
    def _test_op_construction(self, array, op):
        a = op(array)
        v = isle.Vector(op(array))
        self.assertTrue(core.isEqual(a, v),
                        msg="Failed check for scalar type: {typ} and operation: {op}".format(
                            typ=array.dtype, op=op))

    #----------------------------
    def _test_buffer_construction(self, vtyp):
        """
        Test construction from numpy array,
        type checks vector with given size and compares elements
        """

        # basic construction
        typ = self.scalarTypes[vtyp]
        if np.issubdtype(typ, np.signedinteger):
            core.get_logger().warning("Test for buffer construction of integer tensor fails")
            return
        array = rand.randn(RAND_MIN, RAND_MAX, self.Nvec, typ)
        vec = vtyp(array)
        self.assertIsInstance(vec[0], typ,
                              msg="Failed type check for scalar type: {typ} and vector type: {vtyp}".format(
                                  typ=typ, vtyp=vtyp
                              ))
        self.assertTrue(core.isEqual(array, vec),
            msg="Failed equality check for scalar type: {typ} and vector type: {vtyp}".format(
                typ=typ, vtyp=vtyp
            ) + "\npyVec = {pyVec}\ncVec = {cVec}".format(
                pyVec=str(array), cVec=str(vec)
            ))

        # construction after a numpy operation
        # from np operator
        aux = rand.randn(RAND_MIN, RAND_MAX, self.Nvec, typ)
        self._test_op_construction(array, lambda a: a+aux)
        self._test_op_construction(array, lambda a: a*aux)
        self._test_op_construction(array, lambda a: a//aux)

        # from rank 1 array
        self._test_op_construction(array, np.real)
        self._test_op_construction(array, np.imag)
        self._test_op_construction(array, np.abs)
        self._test_op_construction(array, np.exp)
        self._test_op_construction(array, np.sin)
        self._test_op_construction(array, np.cosh)
        self._test_op_construction(array, lambda a: np.roll(a, 5))

        # from rank 2 array
        array = rand.randn(RAND_MIN, RAND_MAX, [self.Nvec]*2, typ)
        self._test_op_construction(array, np.ravel)
        self._test_op_construction(array, lambda a: np.reshape(a, (self.Nvec**2, )))

    #----------------------------
    def _test_list_construction(self, vtyp):
        """
        Test construction from list,
        type checks vector with given size and compares elements
        """
        typ = self.scalarTypes[vtyp]
        pyVec = rand.randn(RAND_MIN, RAND_MAX, self.Nvec, typ)
        cVec  = vtyp(list(pyVec))
        self.assertIsInstance(
            cVec[0],
            typ,
            msg="Failed type check for scalar type: {typ} and vector type: {vtyp}".format(
              typ=typ, vtyp=vtyp
            )
        )
        self.assertTrue(
            core.isEqual(pyVec, cVec),
            msg="Failed equality check for scalar type: {typ} and vector type: {vtyp}".format(
                typ=typ, vtyp=vtyp
            ) + "\npyVec = {pyVec}\ncVec = {cVec}".format(
                pyVec=str(pyVec), cVec=str(cVec)
            )
        )
        return cVec, pyVec
    #----------------------------
    def test_1_construction(self):
        """
        Test size, buffer and list construction of vectors.
        Compares elementwise.
        """
        logger = core.get_logger()
        for vtyp in self.cVecTypes:
            logger.info("Testing constructor of {vtyp}".format(vtyp=vtyp))
            self._test_size_construction(vtyp)
            self._test_buffer_construction(vtyp)
            self._test_list_construction(vtyp)
#-------------- Constructors --------------


#-------------- Operators ------------------
    def test_2_operators(self):
        "Test all operator overloads"
        logger = core.get_logger()
        # iterate operator types: "+", "-", ...
        for opType, operations in self.operations.items():
            # iterate input types: ("vec", "vec"), ("vec", "double"), ...
            for (inTypes, outInstance) in operations:
                logger.info(
                    "Testing: {cIn1:24s} {op:3s} {cIn2:24s} = {cOut:24s}".format(
                        cIn1=str(inTypes[0]), op=str(opType), cIn2=str(inTypes[1]), cOut=str(outInstance)
                    )
                )
                # Generate random c and python input
                if inTypes[0] in self.cVecTypes:
                    cIn1, pyIn1 = self._test_list_construction(inTypes[0])
                else:
                    cIn1 = pyIn1 = rand.randScalar(RAND_MIN, RAND_MAX, inTypes[0])
                if inTypes[1] in self.cVecTypes:
                    cIn2, pyIn2 = self._test_list_construction(inTypes[1])
                else:
                    cIn2 = pyIn2 = rand.randScalar(RAND_MIN, RAND_MAX, inTypes[1])
                # Compute c and python output
                cOut  = core.OperatorDict[opType](cIn1,  cIn2 )
                pyOut = core.OperatorDict[opType](pyIn1, pyIn2)
                # Type check
                self.assertIsInstance(
                    cOut,
                    outInstance,
                    msg= "Type check failed for {cIn1} {op} {cIn2} = {cOut}".format(
                      cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
                    )
                )
                # Value check
                self.assertTrue(
                    core.isEqual(cOut, pyOut),
                    msg="Equality check failed for {cIn1} {op} {cIn2} = {cOut}\n".format(
                        cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
                    )                                           +
                        "cIn1 = {cIn1}\n".format(cIn1=cIn1)     +
                        "cIn2 = {cIn2}\n".format(cIn2=cIn2)     +
                        "cOut = {cOut}\n".format(cOut=cOut)     +
                        "pyOut = {pyOut}\n".format(pyOut=pyOut)
                )
#-------------- Operators ------------------


class TestVector(AbstractVectorTest, unittest.TestCase):
    "Test for all cVec types and opertions"
    Nvec        = 100             # Size of tested vectors
    cVecTypes   = [               # Initializers
        isle.IVector,
        isle.DVector,
        isle.CDVector,
    ]
    scalarTypes = {               # Element type maps
        isle.IVector : int,
        isle.DVector : float,
        isle.CDVector: complex,
    }
    operations = {                # Operations to be tested
      "*"  : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.IVector ),
        ((isle.IVector , int          ), isle.IVector ),
        #((int          , isle.IVector ), isle.IVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        ((isle.IVector , isle.DVector ), isle.DVector ),
        ((isle.DVector,  float        ), isle.DVector ),
        ((isle.DVector,  int          ), isle.DVector ),
        #((float        , isle.DVector ), isle.DVector ),
        #((int        , isle.DVector ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        ((isle.DVector,  isle.CDVector), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
        #((isle.IVector,  isle.CDVector), isle.CDVector),
        ((isle.CDVector, complex      ), isle.CDVector),
        ((isle.CDVector, float        ), isle.CDVector),
        ((isle.CDVector, int          ), isle.CDVector),
        #(( complex     , isle.CDVector), isle.CDVector),
        #(( float       , isle.CDVector), isle.CDVector),
        #(( int         , isle.CDVector), isle.CDVector),
      ],
      "/"  : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.DVector ),
        ((isle.IVector , int          ), isle.DVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        ((isle.IVector , isle.DVector ), isle.DVector ),
        ((isle.DVector,  float        ), isle.DVector ),
        ((isle.DVector,  int          ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        ((isle.DVector,  isle.CDVector), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
        #((isle.IVector,  isle.CDVector), isle.CDVector),
        ((isle.CDVector, complex      ), isle.CDVector),
        ((isle.CDVector, float        ), isle.CDVector),
        ((isle.CDVector, int          ), isle.CDVector),
      ],
      "//"  : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.IVector ),
        ((isle.IVector , int          ), isle.IVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        ((isle.IVector , isle.DVector ), isle.DVector ),
        ((isle.DVector,  float        ), isle.DVector ),
        ((isle.DVector,  int          ), isle.DVector ),
      ],
      "*=" : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.IVector ),
        ((isle.IVector , int          ), isle.IVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        ((isle.DVector,  float        ), isle.DVector ),
        ((isle.DVector,  int          ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
        ((isle.CDVector, complex      ), isle.CDVector),
        ((isle.CDVector, float        ), isle.CDVector),
        ((isle.CDVector, int          ), isle.CDVector),
      ],
      "/=" : [
        # IVector
        #((isle.IVector , isle.IVector ), isle.DVector ),
        #((isle.IVector , int          ), isle.DVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        ((isle.DVector,  float        ), isle.DVector ),
        ((isle.DVector,  int          ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
        ((isle.CDVector, complex      ), isle.CDVector),
        ((isle.CDVector, float        ), isle.CDVector),
        ((isle.CDVector, int          ), isle.CDVector),
      ],
      "+"  : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.IVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        ((isle.IVector , isle.DVector ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        ((isle.DVector,  isle.CDVector), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
        #((isle.IVector,  isle.CDVector), isle.CDVector),
      ],
      "+=" : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.IVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
      ],
      "-"  : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.IVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        ((isle.IVector , isle.DVector ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        ((isle.DVector,  isle.CDVector), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
        #((isle.IVector,  isle.CDVector), isle.CDVector),
      ],
      "-=" : [
        # IVector
        ((isle.IVector , isle.IVector ), isle.IVector ),
        # DVector
        ((isle.DVector , isle.DVector ), isle.DVector ),
        ((isle.DVector , isle.IVector ), isle.DVector ),
        #CVector
        ((isle.CDVector, isle.CDVector), isle.CDVector),
        ((isle.CDVector, isle.DVector ), isle.CDVector),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
      ],
      "@"  : [
        # IVector
        ((isle.IVector , isle.IVector ), int          ),
        # DVector
        ((isle.DVector , isle.DVector ), float        ),
        ((isle.DVector , isle.IVector ), float        ),
        ((isle.IVector , isle.DVector ), float        ),
        #CVector
        ((isle.CDVector, isle.CDVector), complex      ),
        ((isle.CDVector, isle.DVector ), complex      ),
        ((isle.DVector,  isle.CDVector), complex      ),
        #((isle.CDVector, isle.IVector ), isle.CDVector),
        #((isle.IVector,  isle.CDVector), isle.CDVector),
      ],
    }

def setUpModule():
    "Setup the vector test module."
    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    min:  {}
    max:  {}""".format(SEED, RAND_MIN, RAND_MAX))

    rand.setup(SEED)
