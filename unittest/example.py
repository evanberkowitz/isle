#!/usr/bin/env python
"""
Example python module for unittests.
"""
import unittest
try:
  import cns
except ModuleNotFoundError as e:
  print("[WARNING] Please copy the build `cns` module in this directory before running the tests...")
  raise e

import numpy as np
import operator

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
#-------------------------------

sources = [el for el in os.listdir() if ".so" in el]

#-------------------------------

version = str(sub.check_output(["git", "rev-parse", "HEAD"]).strip())


#===============================================================================
#     Helper
#===============================================================================
class RandomInitializer(object):
  "Class for randomly initializin objects"
  def __init__(self):
    "Initialize the type map"
    self.type_map = {
      "int"    : lambda **kwargs : self._init_scalar(int,       **kwargs),
      "float"  : lambda **kwargs : self._init_scalar(float,     **kwargs),
      "double" : lambda **kwargs : self._init_scalar(float,     **kwargs),
      "DVector": lambda **kwargs : self._init_cvec(cns.DVector, **kwargs),
    }

  #----- Initialize a scalar -----
  def _init_scalar(self, typefunc, **kwargs):
    "Initializes a random scalar of type `dtype`"
    val = np.random.uniform()*10
    return typefunc(val), typefunc(val)

  #----- init_cvec -----
  def _init_cvec(self, cvecType, NVec=None):
    """
    Randomly sets all compenents of a vector cvecType.
    Returns the CVec and the NumpyVec.
    """
    # Size of vector
    if NVec is None:
      NVec = int(np.random.uniform()*10+1)
    # create vectors
    pyvec = np.random.uniform(size=NVec)
    cvec  = cvecType(NVec)
    # Set cvec elements
    for nel, el in enumerate(pyvec):
      cvec[nel] = el
    return cvec, pyvec

  #----- Wrapper -----
  def initialize(self, dtype, **kwargs):
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
    if i1 and i2:
      if len(o1) == len(o2):
        for el1, el2 in zip(o1, o2):
          if abs(el1 - el2) > numPrec:
            return False
        return True
      else:
        return False
    elif not(i1) and not(i2):
      return abs(o1 - o2) < numPrec
    else:
      return False


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
class TestDVector(unittest.TestCase):
  "DVector test class"
  #----- Class Properties -----
  cvecType   = cns.DVector
  operations = {
    "*" : [
      # input types            output types
      (("DVector", "DVector"), cns.DVector),
      (("DVector", "int"    ), cns.DVector),
      (("DVector", "double" ), cns.DVector),
      (("DVector", "double" ), cns.DVector),
      (("int"    , "DVector"), cns.DVector),
      (("double" , "DVector"), cns.DVector),
      (("double" , "DVector"), cns.DVector),
    ],
    "/" : [
      (("DVector", "int"),    cns.DVector),
      (("DVector", "double"), cns.DVector),
    ],
    "+" : [
      (("DVector", "DVector"), cns.DVector),
    ],
    "-" : [
      (("DVector", "DVector"), cns.DVector),
    ],
    "@" : [
      (("DVector", "DVector"), float),
    ],
  }

  #----------init test ------------------
  def test_0_init_vec(self):
    "Test if constructor of vector works"
    NVec = int(np.random.uniform()*10+1)
    vec = self.cvecType(NVec)
    self.assertEqual(NVec, len(vec))

  #----------set test ------------------
  def test_1_set_and_get(self):
    "Test setter and getter of components work"
    NVec = int(np.random.uniform()*10+1)
    n1   = int(np.random.uniform()*10)
    v1   = np.random.uniform()
    vec  = self.cvecType(NVec)
    # set random component
    vec[n1] = v1
    self.assertTrue( abs(vec[n1]-v1) < 1.e-7 )

  #----------failing test ------------------
  def test_2_test_operators(self):
    "Test the operator overloads"
    NVec = int(np.random.uniform()*10+1)
    print("\n")
    for opType, operations in self.operations.items():
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
        try:
          self.assertTrue(isinstance(cOut, outInstance))
        except AssertionError as e:
          print("\n")
          logger.info(
            "Type check failed for {cIn1} {op} {cIn2} = {cOut}".format(
              cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
            )
          )
          raise e
        try:
          self.assertTrue(EQ.isEqual(cOut, pyOut))
        except AssertionError as e:
          print("\n")
          logger.info(
            "Equality check failed for {cIn1} {op} {cIn2} = {cOut}".format(
              cIn1=inTypes[0], op=opType, cIn2=inTypes[1], cOut=outInstance
            )
          )
          raise e

#===============================================================================
#     Exe
#===============================================================================
if __name__ == "__main__":
  logger.info("Sources = " + ", ".join(sources) )
  logger.info("Version = " + version )
  logger.info("Starting unittests.\n" )

  unittest.main()