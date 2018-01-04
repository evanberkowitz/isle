#!/usr/bin/env python

"""
Example python module for unittests.
"""

import unittest     # unittest module
import operator     # acces to operators as functions
import abc          # abstract classes
import numpy as np  # numpy!

import core                 # base setup and import
core.prepare_cnxx_import()
import cnxx                 # c++ bindings
import rand                 # random initializer

# RNG params
SEED = 1
RAND_MIN = -1
RAND_MAX = +1



#===============================================================================
#     Abstract basis Test
#===============================================================================
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
    def _test_buffer_construction(self, vtyp):
        """
        Test construction by numpy array, 
        type checks vector with given size and compares elements
        """
        typ = self.scalarTypes[vtyp]
        array = rand.randn(RAND_MIN, RAND_MAX, self.Nvec, typ)
        vec = vtyp(array)
        self.assertIsInstance(
            cVec[0], 
            typ,
            msg="Failed type check for scalar type: {typ} and vector type: {vtyp}".format(
              typ=typ, vtyp=vtyp
            )
        )
        self.assertTrue(
            core.isEqual(array, vec),
            msg="Failed equality check for scalar type: {typ} and vector type: {vtyp}".format(
                typ=typ, vtyp=vtyp
            ) + "\npyVec = {pyVec}\ncVec = {cVec}".format(
                pyVec=str(array), cVec=str(vec)
            )
        )
    #----------------------------
    def _test_list_construction(self, vtyp):
        """
        Test construction by list, 
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
            # self._test_buffer_construction(vtyp, typ)
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
                    "Testing: {cIn1} {op} {cIn2} = {cOut}".format(
                        cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
                    )
                )
                # Generate random c and python input
                if inTypes[0] in self.cVecTypes:
                    cIn1, pyIn1 = self._test_list_construction(inTypes[0])
                else:
                    cIn1, pyIn1 = rand.randScalar(RAND_MIN, RAND_MAX, inTypes[0])
                if inTypes[1] in self.cVecTypes:
                    cIn2, pyIn2 = self._test_list_construction(inTypes[1])
                else:
                    cIn2, pyIn2 = rand.randScalar(RAND_MIN, RAND_MAX, inTypes[1])
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


#===============================================================================
#     Unit Tests
#===============================================================================
class TestVector(AbstractVectorTest, unittest.TestCase):
    "Test for all cVec types and opertions"
    Nvec        = 100             # Size of tested vectors
    cVecTypes   = [               # Initializers
        cnxx.IVector,
        cnxx.DVector,
        cnxx.CDVector,
    ]
    scalarTypes = {               # Element type maps
        cnxx.IVector :  int,
        cnxx.DVector :  float,
        cnxx.CDVector: complex,
    }
    operations = {                # Operations to be tested
      "*" : [
        ((cnxx.DVector , cnxx.DVector ), cnxx.DVector ),
        ((cnxx.IVector , cnxx.IVector ), cnxx.IVector ),
        ((cnxx.CDVector, cnxx.CDVector), cnxx.CDVector),
      ],
    }


#===============================================================================
#     Setup
#===============================================================================
def setUpModule():
    "Setup the vector test module."

    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    min:  {}
    max:  {}""".format(SEED, RAND_MIN, RAND_MAX))

    rand.setup(SEED)


if __name__ == "__main__":
    unittest.main()
