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
#     Tests
#===============================================================================
class TestDVector(unittest.TestCase):
  "DVector test class"
  #----------init test ------------------
  def test_0_init_vec(self):
    "Test if constructor of vector works"
    NVec = int(np.random.uniform()*10+1)
    vec = cns.DVector(NVec)
    self.assertEqual(NVec, len(vec))
  #----------set test ------------------
  def test_1_set_and_get(self):
    "Test setter and getter of components work"
    NVec = int(np.random.uniform()*10+1)
    n1   = int(np.random.uniform()*10)
    v1   = np.random.uniform()
    vec  = cns.DVector(NVec)
    # set random component
    vec[n1] = v1
    self.assertTrue( abs(vec[n1]-v1) < 1.e-7 )
  #----------failing test ------------------
  def test_1_set_and_get(self):
    "To show what happens if it fails"
    self.assertTrue( False )


#===============================================================================
#     Exe
#===============================================================================
if __name__ == "__main__":
  logger.info("Sources = " + ", ".join(sources) )
  logger.info("Version = " + version )
  logger.info("Starting unittests.\n" )

  unittest.main()