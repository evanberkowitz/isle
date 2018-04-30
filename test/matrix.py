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
#     Define matrix iteration
#===============================================================================
def matIterator(mat, dtype=float):
  """
  Creates python objects from the input matrix.
  The input matrix must be cns matrix types.
  """
  isSparse = 'Sparse' in str(type(mat))

  oMat = np.zeros([mat.rows(), mat.columns()], dtype=dtype)

  for nRow in range(mat.rows()):
    for nCol, val in enumerate(mat.row(nRow)):
      if isSparse:
        oMat[nRow, val[0]] = val[1]
      else:
        oMat[nRow, nCol] = val

  return oMat



#===============================================================================
#     Abstract basic test
#===============================================================================
class AbstractMatrixTest(metaclass=abc.ABCMeta):
    "Abstract matrix class test. Matrix test must inherit from this class"
    mShape      = None  # Shape of tested matrix
    matrixTypes = {}    # Matrix types for given scalars
    vectorTypes = {}    # Matrix types for given scalars
    operations  = {}    # Dict for matrix vector multiplications

#-------------- Constructors --------------
    def _test_size_construction(self, mtyp, verbose=False):
        "Test construction by size and type checks vector with given size"
        if verbose:
            logger = core.get_logger()
            logger.info("Testing size construction of {mtyp}".format(mtyp=mtyp))
        typ   = self.matrixTypes[mtyp]
        cMat  = mtyp(*self.mShape)
        # Create symmetric mat if required
        if not("Symmetric" in str(mtyp)):
            pyMat = rand.randn(RAND_MIN, RAND_MAX, self.mShape, typ)
        else:
            pyMat = rand.randn(RAND_MIN, RAND_MAX, [self.mShape[0]]*2, typ)
            pyMat += pyMat.T
        # Set elements
        for nRow, row in enumerate(pyMat):
          for nCol, el in enumerate(row):
            cMat[nRow, nCol] = el

        # Check types
        self.assertIsInstance(
            cMat[0,0],
            typ,
            msg="Failed type check for scalar type: {typ} and matrix type: {mtyp}".format(
              typ=typ, mtyp=mtyp
            )
        )
        # Check elements
        self.assertTrue(
            core.isEqual(pyMat, matIterator(cMat, dtype=typ)),
            msg="Failed equality check matrix type: {mtyp}".format(
                typ=typ, mtyp=mtyp
            ) + "\npyMat = {pyMat}\ncMat = {cMat}".format(
                pyMat=str(pyMat), cMat=str(cMat)
            )
        )
        return cMat, np.matrix(pyMat)

    #----------------------------
    def _test_op_construction(self, array, op):
        a = op(array)
        m = cns.Matrix(op(array))
        self.assertTrue(core.isEqual(a, matIterator(m, dtype=a.dtype)),
                        msg="Failed check for scalar type: {typ} and operation: {op}".format(
                          typ=array.dtype, op=op))

    #----------------------------
    def _test_buffer_construction(self, mtyp, verbose=False):
        """
        Test construction by numpy array,
        type checks matrix with given size and compares elements.
        Ignores integer types.
        """

        typ = self.matrixTypes[mtyp]
        if typ is int: return

        if verbose:
            logger = core.get_logger()
            logger.info("Testing buffer construction of {mtyp}".format(mtyp=mtyp))

        # basic construction
        pyMat = rand.randn(RAND_MIN, RAND_MAX, self.mShape, typ)
        cMat = mtyp(pyMat)
        self.assertIsInstance(
            cMat[0,0],
            typ,
            msg="Failed type check for scalar type: {typ} and matrix type: {mtyp}".format(
              typ=typ, mtyp=mtyp
            )
        )
        self.assertTrue(
            core.isEqual(pyMat, matIterator(cMat, dtype=typ)),
            msg="Failed equality check for matrix type: {mtyp}".format(
                typ=typ, mtyp=mtyp
            ) + "\npyMat = {pyMat}\ncMat = {cMat}".format(
                pyMat=str(pyMat), cMat=str(cMat)
            )
        )

        # construction after a numpy operation
        # from np operator
        aux = rand.randn(RAND_MIN, RAND_MAX, self.mShape, typ)
        self._test_op_construction(pyMat, lambda a: a+aux)
        self._test_op_construction(pyMat, lambda a: a*aux)
        self._test_op_construction(pyMat, lambda a: a//aux)

        # from rank 2 array
        self._test_op_construction(pyMat, np.real)
        self._test_op_construction(pyMat, np.imag)
        self._test_op_construction(pyMat, np.abs)
        self._test_op_construction(pyMat, np.exp)
        self._test_op_construction(pyMat, np.sin)
        self._test_op_construction(pyMat, np.cosh)
        self._test_op_construction(pyMat, lambda a: np.roll(a, 5))
        self._test_op_construction(pyMat, np.transpose)
        self._test_op_construction(pyMat, lambda a: np.moveaxis(a, 0, -1))

        # from rank 1 array
        vec = rand.randn(RAND_MIN, RAND_MAX, self.mShape[0]*self.mShape[1], typ)
        self._test_op_construction(vec, lambda a: np.reshape(a, self.mShape))

    #----------------------------
    def _test_list_construction(self, mtyp, verbose=False, transpose=False):
        """
        Test construction by list,
        type checks matrix with given size and compares elements
        """
        logger = core.get_logger()
        if verbose: logger.info("Testing list construction of {mtyp}".format(mtyp=mtyp))
        typ = self.matrixTypes[mtyp]
        pyMat = []
        mat   = rand.randn(RAND_MIN, RAND_MAX, self.mShape, typ)
        if transpose: mat = mat.T
        for row in mat:
          pyMat += [[el for el in row]]
        cMat = mtyp(pyMat)

        self.assertIsInstance(
            cMat[0,0],
            typ,
            msg="Failed type check for scalar type: {typ} and matrix type: {mtyp}".format(
              typ=typ, mtyp=mtyp
            )
        )
        self.assertTrue(
            core.isEqual(pyMat, matIterator(cMat, dtype=typ)),
            msg="Failed equality check matrix type: {mtyp}".format(
                typ=typ, mtyp=mtyp
            ) + "\npyMat = {pyMat}\ncMat = {cMat}".format(
                pyMat=str(pyMat), cMat=str(cMat)
            )
        )
        return cMat, np.matrix(pyMat)
    #----------------------------
    def test_1_construction(self):
        """
        Test size, buffer and list construction of matrices.
        Compares elementwise.
        """
        logger = core.get_logger()
        for mtyp in self.matrixTypes.keys():
            logger.info("Testing constructor of {mtyp}".format(mtyp=mtyp))
            self._test_size_construction(mtyp, verbose=True)
            if not("Sparse" in str(mtyp)):
              self._test_buffer_construction(mtyp, verbose=True)
              self._test_list_construction(mtyp, verbose=True)
#-------------- Constructors --------------

#-------------- Buffer Protocol --------------
    def test_2_buffer_protocol(self):
        """
        Test the buffer protocol on matrices.
        """
        logger = core.get_logger()
        for mtyp in self.matrixTypes:
            logger.info("Testing buffer protocol for %s", mtyp)
            typ = self.matrixTypes[mtyp]
            if typ is int or "Sparse" in str(mtyp): continue

            pyMat = rand.randn(RAND_MIN, RAND_MAX, self.mShape, typ)
            cMat = mtyp(pyMat)
            self.assertTrue(
                core.isEqual(pyMat, np.array(cMat, copy=False)),
                msg=f"Failed equality check for matrix type: {mtyp}"
                + f"\npyMat = {pyMat}\ncMat = {cMat}"
            )
#-------------- Buffer Protocol --------------

#-------------- Vector Constructors --------------
    def _construct_vector(self, vtyp, transpose=False):
        """
        Constructs a vector which is valid for matrix multiplication
        """
        typ = self.vectorTypes[vtyp]
        size = self.mShape[0] if transpose else self.mShape[1]
        pyVec = rand.randn(RAND_MIN, RAND_MAX, size, typ)
        cVec  = vtyp(list(pyVec))
        return cVec, np.matrix(pyVec).T if transpose else np.matrix(pyVec)


#-------------- Operators ------------------
    def test_3_operators(self):
        "Test all operator overloads"
        logger = core.get_logger()
        # iterate operator types: "+", "-", ...
        for opType, operations in self.operations.items():
            # iterate input types: ("mat", "mat"), ("mat", "double"), ...
            for (inTypes, outInstance) in operations:
                logger.info(
                    "Testing: {cIn1:24s} {op:3s} {cIn2:24s} = {cOut:24s}".format(
                        cIn1=str(inTypes[0]), op=str(opType), cIn2=str(inTypes[1]), cOut=str(outInstance)
                    )
                )
                # Generate random c and python input
                if inTypes[0] in self.matrixTypes.keys():
                    cIn1, pyIn1 = self._test_list_construction(inTypes[0])
                elif inTypes[0] in self.vectorTypes.keys():
                    cIn1, pyIn1 = self._construct_vector(inTypes[0])
                else:
                    cIn1 = pyIn1 = rand.randScalar(RAND_MIN, RAND_MAX, inTypes[0])

                if inTypes[1] in self.matrixTypes.keys():
                    cIn2, pyIn2 = self._test_list_construction(inTypes[1], transpose=True)
                elif inTypes[1] in self.vectorTypes.keys():
                    cIn2, pyIn2 = self._construct_vector(inTypes[1], transpose=True)
                else:
                    cIn2 = pyIn2 = rand.randScalar(RAND_MIN, RAND_MAX, inTypes[1])

                # Compute c and python output
                cOut  = core.OperatorDict[opType](cIn1,  cIn2 )
                pyOut = np.array(core.OperatorDict[opType](pyIn1, pyIn2))


                # Type check matrixTypes
                self.assertIsInstance(
                    cOut,
                    outInstance,
                    msg= "Type check failed for {cIn1} {op} {cIn2} = {cOut}".format(
                      cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
                    )
                )
                # Value check
                ## Check type for casting to iterable
                if type(cOut) in self.matrixTypes.keys():
                    cOut = matIterator(cOut, dtype=self.matrixTypes[type(cOut)])
                # Cast matrix back to array
                elif type(cOut) in self.vectorTypes.keys():
                    pyOut = np.array(pyOut).T[0]
 
                ## Check value
                self.assertTrue(
                    core.isEqual(
                      cOut,
                      pyOut,
                      nOps=self.mShape[0] if "*" in opType else 1
                    ),
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
#     Unittest for matrices --- needs to be further extended
#===============================================================================
class TestMatrix(AbstractMatrixTest, unittest.TestCase):
    "Test for all cVec types and opertions"
    mShape      = (40, 40)        # Shape of tested matrix
    matrixTypes = {               # Element type maps
        cns.IMatrix: int,
        cns.ISparseMatrix: int,
        #cns.ISymmetricSparseMatrix: int,
        cns.DMatrix: float,
        cns.DSparseMatrix: float,
        #cns.DSymmetricSparseMatrix: float,
        cns.CDMatrix: complex,
        cns.CDSparseMatrix: complex,
        #cns.CDSymmetricSparseMatrix: complex,
    }
    vectorTypes = {
        cns.IVector : int,
        cns.DVector : float,
        cns.CDVector: complex,
    }
    operations = {
      "*"  : [
        # IMatrix
        ((cns.IMatrix , cns.IMatrix ), cns.IMatrix ),
        ((cns.IMatrix , int         ), cns.IMatrix ),
        # IVector
        ((cns.IMatrix , cns.IVector ), cns.IVector ),
        # DMatrix
        ((cns.DMatrix , cns.DMatrix ), cns.DMatrix ),
        ((cns.DMatrix , cns.IMatrix ), cns.DMatrix ),
        ((cns.IMatrix , cns.DMatrix ), cns.DMatrix ),
        ((cns.DMatrix,  float       ), cns.DMatrix ),
        ((cns.DMatrix,  int         ), cns.DMatrix ),
        # DVector
        ((cns.DMatrix , cns.DVector ), cns.DVector ),
        ((cns.DMatrix , cns.IVector ), cns.DVector ),
        ((cns.IMatrix , cns.DVector ), cns.DVector ),
        # CMatrix
        ((cns.CDMatrix, cns.CDMatrix), cns.CDMatrix),
        ((cns.CDMatrix, cns.DMatrix ), cns.CDMatrix),
        ((cns.DMatrix,  cns.CDMatrix), cns.CDMatrix),
        ((cns.CDMatrix, complex      ), cns.CDMatrix),
        ((cns.CDMatrix, float        ), cns.CDMatrix),
        ((cns.CDMatrix, int          ), cns.CDMatrix),
        # CVector
        ((cns.CDMatrix, cns.CDVector), cns.CDVector),
        ((cns.CDMatrix, cns.DVector ), cns.CDVector),
        ((cns.DMatrix , cns.CDVector), cns.CDVector),
      ],
      "*=" : [
        # IMatrix
        ((cns.IMatrix , cns.IMatrix ), cns.IMatrix ),
        ((cns.IMatrix , int          ), cns.IMatrix ),
        # DMatrix
        ((cns.DMatrix , cns.DMatrix ), cns.DMatrix ),
        ((cns.DMatrix , cns.IMatrix ), cns.DMatrix ),
        ((cns.DMatrix,  float        ), cns.DMatrix ),
        ((cns.DMatrix,  int          ), cns.DMatrix ),
        #CMatrix
        ((cns.CDMatrix, cns.CDMatrix), cns.CDMatrix),
        ((cns.CDMatrix, cns.DMatrix ), cns.CDMatrix),
        #((cns.CDMatrix, cns.IMatrix ), cns.CDMatrix),
        ((cns.CDMatrix, complex      ), cns.CDMatrix),
        ((cns.CDMatrix, float        ), cns.CDMatrix),
        ((cns.CDMatrix, int          ), cns.CDMatrix),
      ],
      "+"  : [
        # IMatrix
        ((cns.IMatrix , cns.IMatrix ), cns.IMatrix ),
        # DMatrix
        ((cns.DMatrix , cns.DMatrix ), cns.DMatrix ),
        ((cns.DMatrix , cns.IMatrix ), cns.DMatrix ),
        ((cns.IMatrix , cns.DMatrix ), cns.DMatrix ),
        #CMatrix
        ((cns.CDMatrix, cns.CDMatrix), cns.CDMatrix),
        ((cns.CDMatrix, cns.DMatrix ), cns.CDMatrix),
        ((cns.DMatrix,  cns.CDMatrix), cns.CDMatrix),
        #((cns.CDMatrix, cns.IMatrix ), cns.CDMatrix),
        #((cns.IMatrix,  cns.CDMatrix), cns.CDMatrix),
      ],
      "+=" : [
        # IMatrix
        ((cns.IMatrix , cns.IMatrix ), cns.IMatrix ),
        # DMatrix
        ((cns.DMatrix , cns.DMatrix ), cns.DMatrix ),
        ((cns.DMatrix , cns.IMatrix ), cns.DMatrix ),
        #CMatrix
        ((cns.CDMatrix, cns.CDMatrix), cns.CDMatrix),
        ((cns.CDMatrix, cns.DMatrix ), cns.CDMatrix),
        #((cns.CDMatrix, cns.IMatrix ), cns.CDMatrix),
      ],
      "-"  : [
        # IMatrix
        ((cns.IMatrix , cns.IMatrix ), cns.IMatrix ),
        # DMatrix
        ((cns.DMatrix , cns.DMatrix ), cns.DMatrix ),
        ((cns.DMatrix , cns.IMatrix ), cns.DMatrix ),
        ((cns.IMatrix , cns.DMatrix ), cns.DMatrix ),
        #CMatrix
        ((cns.CDMatrix, cns.CDMatrix), cns.CDMatrix),
        ((cns.CDMatrix, cns.DMatrix ), cns.CDMatrix),
        ((cns.DMatrix,  cns.CDMatrix), cns.CDMatrix),
        #((cns.CDMatrix, cns.IMatrix ), cns.CDMatrix),
        #((cns.IMatrix,  cns.CDMatrix), cns.CDMatrix),
      ],
      "-=" : [
        # IMatrix
        ((cns.IMatrix , cns.IMatrix ), cns.IMatrix ),
        # DMatrix
        ((cns.DMatrix , cns.DMatrix ), cns.DMatrix ),
        ((cns.DMatrix , cns.IMatrix ), cns.DMatrix ),
        #CMatrix
        ((cns.CDMatrix, cns.CDMatrix), cns.CDMatrix),
        ((cns.CDMatrix, cns.DMatrix ), cns.CDMatrix),
        #((cns.CDMatrix, cns.IMatrix ), cns.CDMatrix),
      ],
    }



#===============================================================================
#     Setup
#===============================================================================
def setUpModule():
    "Setup the matrix test module."

    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    min:  {}
    max:  {}""".format(SEED, RAND_MIN, RAND_MAX))

    rand.setup(SEED)


if __name__ == "__main__":
    unittest.main()
