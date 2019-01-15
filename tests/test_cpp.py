#!/usr/bin/env python

"""
Compiled C++ internal unittests
"""

# import unittest
# import subprocess
# from pathlib import Path

# from . import core

# EXE = Path(__file__).parent/'bin'/'isle_cpp_test'

# class TestCpp(unittest.TestCase):
#     """Runner for compiled tests of C++ library internals."""

#     def test_1_cpp(self):
#         "Run the binary"
#         logger = core.get_logger()
#         print()
#         logger.info('Testing C++ internals')
#         rc = subprocess.run([str(EXE)]).returncode
#         self.assertEqual(rc, 0, 'C++ internal tests did not finish successfully')
