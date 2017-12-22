#!/usr/bin/env python
"""
Base implementation of unittests
"""
try:
  import cns
except ModuleNotFoundError as e:
  print("[WARNING] Please copy the build `cns` module in this directory before running the tests...")
  raise e

import numpy as np
import operator
import abc

#===============================================================================
#     logging
#===============================================================================
import os
import subprocess as sub
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
_format = logging.Formatter(
  '[%(asctime)s] %(message)s',
  "%Y-%m-%d %H:%M"
)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_format)

logger.addHandler(_ch)


#===============================================================================
#     Helper
#===============================================================================
class RandomInitializer(object):
  "Class for randomly initializin objects"
  def __init__(self):
    "Initialize the type map"
    self.type_map = {
      "int"      : lambda **kwargs : self._init_scalar(int,       **kwargs),
      "float"    : lambda **kwargs : self._init_scalar(float,     **kwargs),
      "double"   : lambda **kwargs : self._init_scalar(float,     **kwargs),
      "DVector"  : lambda **kwargs : self._init_cvec(cns.DVector, dtype=float, **kwargs),
      "IVector"  : lambda **kwargs : self._init_cvec(cns.IVector, dtype=int, **kwargs),
      cns.DVector: lambda **kwargs : self._init_cvec(cns.DVector, dtype=float, **kwargs),
      cns.IVector: lambda **kwargs : self._init_cvec(cns.IVector, dtype=int, **kwargs),
    }

  #----- Initialize a scalar -----
  def _init_scalar(self, typefunc, **kwargs):
    "Initializes a random scalar of type `dtype`"
    val = np.random.uniform()*10+1
    return typefunc(val), typefunc(val)

  #----- init_cvec -----
  def _init_cvec(self, cvecType, NVec=None, dtype=float):
    """
    Randomly sets all compenents of a vector cvecType.
    Returns the CVec and the NumpyVec.
    """
    # Size of vector
    if NVec is None:
      NVec = int(np.random.uniform()*10+1)
    # create vectors
    pyvec = np.array(np.random.uniform(size=NVec)*5-10, dtype=dtype)
    cvec  = cvecType(NVec)
    # Set cvec elements
    for nel, el in enumerate(pyvec):
      cvec[nel] = el
    return cvec, pyvec

  #----- Wrapper -----
  def initialize(self, dtype, **kwargs):
    """
    Wrapper for initializing objects.
    Always returns c++ and python object for comparism.
    """
    out = self.type_map.get(dtype)(**kwargs)
    if out is None:
      raise TypeError("Do not know how to initialize: '{t}'".format(t=dtype))
    return out


class Equalizer(object):
  "Class needed to compare two different objects"
  def isEqual(self, o1, o2, numPrec=1.e-7):
    "If iterable compares element wise, if not direct equal or fail"
    i1 = hasattr(o1 ,"__iter__")
    i2 = hasattr(o2 ,"__iter__")
    # test if both iterable
    if i1 and i2: # both iterable
      if len(o1) == len(o2): # same length
        for el1, el2 in zip(o1, o2): # element wise checks
          if abs(el1 - el2) > numPrec:
            return False
        return True
      else: # not same length
        return False
    elif not(i1) and not(i2): # both not iterable
      return abs(o1 - o2) < numPrec
    else: # one iterable not iterable
      return False

# Easy way to acces operators as functions
OperatorDict = {
  '+'  : operator.add,
  '+=' : operator.iadd,
  '-'  : operator.sub,
  '-=' : operator.isub,
  '*'  : operator.mul,
  '*=' : operator.imul,
  '/'  : operator.truediv,
  '/=' : operator.itruediv,
  '@'  : operator.matmul
}

RI = RandomInitializer()
EQ = Equalizer()

#===============================================================================
#     Tests
#===============================================================================
class AbstractVectorTest(metaclass=abc.ABCMeta):
  "Abstract vector class test. Vector test must inherit from this class"
  # C++ base type
  cvecType   = None
  # to be checked operator overloads
  # { operation: ( (InputType1, InputType2), outPutClas ), ...}
  operations = {}

  #----------init test ------------------
  def test_0_init_vec(self):
    "Test if constructor of vector works"
    logger.info("Testing initialization of {vecType}".format(vecType=self.cvecType))
    NVec = int(np.random.uniform()*10+1)
    vec = self.cvecType(NVec)
    self.assertEqual(NVec, len(vec))

  #----------set test ------------------
  def test_1_set_and_get(self):
    "Test setter and getter of components work"
    logger.info("Testing setter of {vecType}".format(vecType=self.cvecType))
    cOut, pyOut = RI.initialize(self.cvecType)
    try:
      self.assertTrue(EQ.isEqual(cOut, pyOut))
    except AssertionError as e:
      logger.info("Equality check failed for cVec = pyVec")
      logger.info("cOut  = {cOut}".format(cOut=cOut))
      logger.info("pyOut = {pyOut}".format(pyOut=pyOut))
      raise e

  #----------failing test ------------------
  def test_2_test_operators(self):
    "Test the operator overloads"
    NVec = int(np.random.uniform()*10+1)
    # iterate operator types: "+", "-", ...
    for opType, operations in self.operations.items():
      # iterate input types: ("vec", "vec"), ("vec", "double"), ...
      for (inTypes, outInstance) in operations:
        logger.info(
          "Testing: {cIn1} {op} {cIn2} = {cOut}".format(
            cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
          )
        )
        cIn1, pyIn1 = RI.initialize(inTypes[0], NVec=NVec)
        cIn2, pyIn2 = RI.initialize(inTypes[1], NVec=NVec)
        cOut  = OperatorDict[opType](cIn1,  cIn2 )
        pyOut = OperatorDict[opType](pyIn1, pyIn2)
        # Type check
        try:
          self.assertTrue(isinstance(cOut, outInstance))
        except AssertionError as e:
          logger.info(
            "Type check failed for {cIn1} {op} {cIn2} = {cOut}".format(
              cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
            )
          )
          raise e
        # Value check
        try:
          self.assertTrue(EQ.isEqual(cOut, pyOut))
        except AssertionError as e:
          logger.info(
            "Equality check failed for {cIn1} {op} {cIn2} = {cOut}".format(
              cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
            )
          )
          logger.info("cIn1 = {cIn1}".format(cIn1=cIn1))
          logger.info("cIn2 = {cIn2}".format(cIn2=cIn2))
          logger.info("cOut = {cOut}".format(cOut=cOut))
          logger.info("pyOut = {pyOut}".format(pyOut=pyOut))
          raise e

