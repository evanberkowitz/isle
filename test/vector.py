#!/usr/bin/env python

"""
Unittest for 'cnxx' vector classes.
"""

import unittest     # unittest module
import operator     # acces to operators as functions
import abc          # abstract classes
import numpy as np  # numpy!

import core                 # base setup and import
core.prepare_module_import()
import cns                  # c++ bindings
import rand                 # random initializer

# RNG params
SEED = 1
RAND_MIN = -5
RAND_MAX = +5



#===============================================================================
#     Abstract basic test
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


#===============================================================================
#     Unittest for vectors  --- needs to be further extended
#===============================================================================
class TestVector(AbstractVectorTest, unittest.TestCase):
    "Test for all cVec types and opertions"
    Nvec        = 100             # Size of tested vectors
    cVecTypes   = [               # Initializers
        cns.IVector,
        cns.DVector,
        cns.CDVector,
    ]
    scalarTypes = {               # Element type maps
        cns.IVector : int,
        cns.DVector : float,
        cns.CDVector: complex,
    }
    operations = {                # Operations to be tested
      "*"  : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.IVector ),
        ((cns.IVector , int          ), cns.IVector ),
        #((int          , cns.IVector ), cns.IVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        ((cns.IVector , cns.DVector ), cns.DVector ),
        ((cns.DVector,  float        ), cns.DVector ),
        ((cns.DVector,  int          ), cns.DVector ),
        #((float        , cns.DVector ), cns.DVector ),
        #((int        , cns.DVector ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        ((cns.DVector,  cns.CDVector), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
        #((cns.IVector,  cns.CDVector), cns.CDVector),
        ((cns.CDVector, complex      ), cns.CDVector),
        ((cns.CDVector, float        ), cns.CDVector),
        ((cns.CDVector, int          ), cns.CDVector),
        #(( complex     , cns.CDVector), cns.CDVector),
        #(( float       , cns.CDVector), cns.CDVector),
        #(( int         , cns.CDVector), cns.CDVector),
      ],
      "/"  : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.DVector ),
        ((cns.IVector , int          ), cns.DVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        ((cns.IVector , cns.DVector ), cns.DVector ),
        ((cns.DVector,  float        ), cns.DVector ),
        ((cns.DVector,  int          ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        ((cns.DVector,  cns.CDVector), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
        #((cns.IVector,  cns.CDVector), cns.CDVector),
        ((cns.CDVector, complex      ), cns.CDVector),
        ((cns.CDVector, float        ), cns.CDVector),
        ((cns.CDVector, int          ), cns.CDVector),
      ],
      "//"  : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.IVector ),
        ((cns.IVector , int          ), cns.IVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        ((cns.IVector , cns.DVector ), cns.DVector ),
        ((cns.DVector,  float        ), cns.DVector ),
        ((cns.DVector,  int          ), cns.DVector ),
      ],
      "*=" : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.IVector ),
        ((cns.IVector , int          ), cns.IVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        ((cns.DVector,  float        ), cns.DVector ),
        ((cns.DVector,  int          ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
        ((cns.CDVector, complex      ), cns.CDVector),
        ((cns.CDVector, float        ), cns.CDVector),
        ((cns.CDVector, int          ), cns.CDVector),
      ],
      "/=" : [
        # IVector
        #((cns.IVector , cns.IVector ), cns.DVector ),
        #((cns.IVector , int          ), cns.DVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        ((cns.DVector,  float        ), cns.DVector ),
        ((cns.DVector,  int          ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
        ((cns.CDVector, complex      ), cns.CDVector),
        ((cns.CDVector, float        ), cns.CDVector),
        ((cns.CDVector, int          ), cns.CDVector),
      ],
      "+"  : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.IVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        ((cns.IVector , cns.DVector ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        ((cns.DVector,  cns.CDVector), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
        #((cns.IVector,  cns.CDVector), cns.CDVector),
      ],
      "+=" : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.IVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
      ],
      "-"  : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.IVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        ((cns.IVector , cns.DVector ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        ((cns.DVector,  cns.CDVector), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
        #((cns.IVector,  cns.CDVector), cns.CDVector),
      ],
      "-=" : [
        # IVector
        ((cns.IVector , cns.IVector ), cns.IVector ),
        # DVector
        ((cns.DVector , cns.DVector ), cns.DVector ),
        ((cns.DVector , cns.IVector ), cns.DVector ),
        #CVector
        ((cns.CDVector, cns.CDVector), cns.CDVector),
        ((cns.CDVector, cns.DVector ), cns.CDVector),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
      ],
      "@"  : [
        # IVector
        ((cns.IVector , cns.IVector ), int          ),
        # DVector
        ((cns.DVector , cns.DVector ), float        ),
        ((cns.DVector , cns.IVector ), float        ),
        ((cns.IVector , cns.DVector ), float        ),
        #CVector
        ((cns.CDVector, cns.CDVector), complex      ),
        ((cns.CDVector, cns.DVector ), complex      ),
        ((cns.DVector,  cns.CDVector), complex      ),
        #((cns.CDVector, cns.IVector ), cns.CDVector),
        #((cns.IVector,  cns.CDVector), cns.CDVector),
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
