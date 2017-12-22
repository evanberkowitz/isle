#!/usr/bin/env python
"""
Example python module for unittests.
"""
import unittest
import baseTest
try:
  import cns
except ModuleNotFoundError as e:
  print("[WARNING] Please copy the build `cns` module in this directory before running the tests...")
  raise e

#===============================================================================
#     logging
#===============================================================================
import os
import subprocess as sub
import logging
logger = logging.getLogger("baseTest.py")
#-------------------------------
sources = [el for el in os.listdir() if ".so" in el]
version = str(sub.check_output(["git", "rev-parse", "HEAD"]).strip())
#-------------------------------


#===============================================================================
#     Unit Tests
#===============================================================================
class TestDVector(baseTest.AbstractVectorTest, unittest.TestCase):
  "DVector test class"
  # C++ base type
  cvecType   = cns.DVector
  # to be checked operator overloads
  # { operation: ( (InputType1, InputType2), outPutClas ), ...}
  operations = {
    "*" : [
      (("DVector", "DVector"), cns.DVector),
      (("DVector", "int"    ), cns.DVector),
      (("DVector", "double" ), cns.DVector),
      (("int"    , "DVector"), cns.DVector),
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


#-------------------------------
class TestIVector(baseTest.AbstractVectorTest, unittest.TestCase):
  "IVector test class"
  # C++ base type
  cvecType   = cns.IVector
  # to be checked operator overloads
  # { operation: ( (InputType1, InputType2), outPutClas ), ...}
  operations = {
    "*" : [
      (("IVector", "IVector"), cns.IVector),
      (("IVector", "int"    ), cns.IVector),
      (("IVector", "double" ), cns.DVector),
      (("int"    , "IVector"), cns.IVector),
      (("double" , "IVector"), cns.DVector),
    ],
   # "/" : [
   #   (("IVector", "int"),    cns.IVector),
   #   (("IVector", "double"), cns.DVector),
   # ],
    "+" : [
      (("IVector", "IVector"), cns.IVector),
    ],
    "-" : [
      (("IVector", "IVector"), cns.IVector),
    ],
    "@" : [
      (("IVector", "IVector"), int),
    ],
  }

#===============================================================================
#     Exe
#===============================================================================
if __name__ == "__main__":
  logger.info("Sources = " + ", ".join(sources) )
  logger.info("Version = " + version )
  logger.info("Starting unittests.\n" )
  unittest.main()