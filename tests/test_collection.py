"""
Unittest for collection module.
"""

from logging import getLogger
import unittest
import random

import isle


# RNG params
SEED = 1875
RAND_MEAN = 0
RAND_STD = 1
MIN_LEN = 0
MAX_LEN = 30

def _randomList():
    return [random.gauss(RAND_MEAN, RAND_STD) for _ in range(random.randint(MIN_LEN, MAX_LEN))]

def _randomSlices(n):
    a = random.randint(0, n)
    b = random.randint(a, n) if random.randint(0, 1) == 0 else None
    s = random.randint(1, max(n//2, 1))
    large = slice(a, b, s)

    if large.stop is None:
        a = large.start+random.randint(0, n//large.step)*large.step
        b = None
    else:
        a = large.start+random.randint(0, (large.stop-large.start)//large.step)*large.step
        b = random.randint(a, large.stop)
    s = random.randint(1, max(n//large.step, 1))*large.step
    small = slice(a, b, s)

    return large, small

class TestCollection(unittest.TestCase):
    def test_1_subslice(self):
        """Test function subslice."""

        for i in range(10000):
            fullList = _randomList()
            large, small = _randomSlices(len(fullList))
            subList = fullList[large]

            subslice = isle.collection.subslice(large, small)
            self.assertEqual(subList[subslice], fullList[small],
                             msg=f"Failed check of subslice in repetition {i}\n"
                             f"with large = {large}, small = {small}, subslice = {subslice}")

def setUpModule():
    "Setup the SumAction test module."

    logger = getLogger(__name__)
    logger.info("""Parameters for RNG:
    seed: {}
    mean: {}
    std:  {}""".format(SEED, RAND_MEAN, RAND_STD))
    random.seed(SEED)
