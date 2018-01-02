#!/usr/bin/env python
"""
Example python module for unittests.
"""
import unittest
import baseTest
try:
  import core
  core.prepare_cnxx_import()
  import cnxx
except ModuleNotFoundError as e:
  print("[WARNING] Please copy the build `cnxx` module in this directory before running the tests...")
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
  cvecType   = cnxx.DVector
  # to be checked operator overloads
  # { operation: ( (InputType1, InputType2), outPutClass ), ...}
  operations = {
    "*" : [
      (("DVector", "DVector"), cnxx.DVector),
      (("DVector", "int"    ), cnxx.DVector),
      (("DVector", "double" ), cnxx.DVector),
      (("int"    , "DVector"), cnxx.DVector),
      (("double" , "DVector"), cnxx.DVector),
    ],
    "/" : [
      (("DVector", "int"),    cnxx.DVector),
      (("DVector", "double"), cnxx.DVector),
    ],
    "+" : [
      (("DVector", "DVector"), cnxx.DVector),
    ],
    "-" : [
      (("DVector", "DVector"), cnxx.DVector),
    ],
    "@" : [
      (("DVector", "DVector"), float),
    ],
  }


#-------------------------------
class TestIVector(baseTest.AbstractVectorTest, unittest.TestCase):
  "IVector test class"
  # C++ base type
  cvecType   = cnxx.IVector
  # to be checked operator overloads
  # { operation: ( (InputType1, InputType2), outPutClass ), ...}
  operations = {
    "*" : [
      (("IVector", "IVector"), cnxx.IVector),
      (("IVector", "int"    ), cnxx.IVector),
      (("IVector", "double" ), cnxx.DVector),
      (("int"    , "IVector"), cnxx.IVector),
      (("double" , "IVector"), cnxx.DVector),
    ],
   # "/" : [
   #   (("IVector", "int"),    cnxx.IVector),
   #   (("IVector", "double"), cnxx.DVector),
   # ],
    "+" : [
      (("IVector", "IVector"), cnxx.IVector),
    ],
    "-" : [
      (("IVector", "IVector"), cnxx.IVector),
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