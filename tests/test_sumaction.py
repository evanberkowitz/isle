#!/usr/bin/env python

"""
Unittest for SumAction.
"""

import unittest
from logging import getLogger

import numpy as np

import isle
from . import core, rand


# RNG params
SEED = 8613
RAND_MEAN = 0
RAND_STD = 0.2
N_REP = 10 # number of repetitions

#
# TODO Actions implemented in Python are not supported anymore
#
class _DummyAction(isle.action.Action):
    def eval(self, phi):
        return np.sum(phi)

    def force(self, phi):
        return phi

def _randomPhi(n):
    "Return a normally distributed random complex vector of n elements."
    real = np.random.normal(RAND_MEAN, RAND_STD, n)
    imag = np.random.normal(RAND_MEAN, RAND_STD, n)
    return isle.Vector(real + 1j*imag)


def _testConstructionPureCXX():
    logger = getLogger(__name__)
    logger.info("Testing construction of SumAction from C++ actions")

    # anonymous variable, init method
    sact = isle.action.SumAction(isle.action.HubbardGaugeAction(1))
    del sact

    # named variable
    act = isle.action.HubbardGaugeAction(1)
    sact = isle.action.SumAction(act)
    del act
    del sact

    # via add method
    sact = isle.action.SumAction()
    sact.add(isle.action.HubbardGaugeAction(1))
    del sact

    # via both methods
    act = isle.action.HubbardGaugeAction(1)
    sact = isle.action.SumAction(isle.action.HubbardGaugeAction(2))
    sact.add(act)
    del sact
    del act

    # via + operator
    act1 = isle.action.HubbardGaugeAction(1)
    act2 = isle.action.HubbardGaugeAction(2)
    sact = act1 + act2
    del act1
    del act2
    del sact

def _testConstructionPurePy():
    logger = getLogger(__name__)
    logger.info("Testing construction of SumAction from Python actions")

    # anonymous variable, init method
    sact = isle.action.SumAction(_DummyAction())
    del sact

    # named variable
    act = _DummyAction()
    sact = isle.action.SumAction(act)
    del act
    del sact

    # via add method
    sact = isle.action.SumAction()
    sact.add(_DummyAction())
    del sact

    # via both methods
    act = _DummyAction()
    sact = isle.action.SumAction(_DummyAction())
    sact.add(act)
    del sact
    del act

    # via + operator
    act1 = _DummyAction()
    act2 = _DummyAction()
    sact = act1 + act2
    del act1
    del act2
    del sact

def _testConstructionMixed():
    logger = getLogger(__name__)
    logger.info("Testing construction of action.SumAction from C++ and Python actions")

    # anonymous variable, init method
    sact = isle.action.SumAction(_DummyAction(), isle.action.HubbardGaugeAction(1))
    del sact

    # named variable
    act = _DummyAction()
    sact = isle.action.SumAction(isle.action.HubbardGaugeAction(1), act)
    del act
    del sact

    # via add method
    sact = isle.action.SumAction()
    sact.add(_DummyAction())
    sact.add(isle.action.HubbardGaugeAction(1))
    del sact

    # via both methods
    act = _DummyAction()
    sact = isle.action.SumAction(_DummyAction())
    sact.add(act)
    del act
    sact.add(isle.action.HubbardGaugeAction(1))
    act = isle.action.HubbardGaugeAction(1)
    sact.add(act)
    del sact
    del act

    # via + operator
    act1 = _DummyAction()
    act2 = _DummyAction()
    sact = act1 + act2 + isle.action.HubbardGaugeAction(3)
    del act1
    del act2
    del sact


class TestSumAction(unittest.TestCase):
    def test_1_construction(self):
        "Test construction (and tear down) of isle.action.SumAction."
        # No assertions here but if it completes without a segfault, it should be fine.
        _testConstructionPureCXX()
        # _testConstructionPurePy()
        # _testConstructionMixed()

    def test_2_eval(self):
        "Test eval function of isle.action.SumAction."

        logger = getLogger(__name__)
        logger.info("Testing SumAction.eval()")
        for rep in range(N_REP):
            sact = isle.action.SumAction(isle.action.HubbardGaugeAction(1))
            phi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                  + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)

            sacteval = sact.eval(isle.Vector(phi))
            manualeval = np.dot(phi, phi)/2

            self.assertAlmostEqual(sacteval, manualeval, places=10,
                                   msg="Failed check of SumAction.eval in repetition {}\n".format(rep)\
                                   + f"with sacteval = {sacteval}, manualeval = {manualeval}")

    def test_3_force(self):
        "Test force function of isle.action.SumAction."

        logger = getLogger(__name__)
        logger.info("Testing SumAction.force()")
        for rep in range(N_REP):
            sact = isle.action.SumAction(isle.action.HubbardGaugeAction(1))
            phi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                  + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)

            sactforce = sact.force(isle.Vector(phi))
            manualforce = -phi/1

            self.assertAlmostEqual(np.linalg.norm(sactforce-manualforce), 0., places=10,
                                   msg="Failed check of SumAction.force in repetition {}\n".format(rep)\
                                   + f"with sactforce = {sactforce}, manualforce = {manualforce}")

def setUpModule():
    "Setup the SumAction test module."

    logger = getLogger(__name__)
    logger.info("""Parameters for RNG:
    seed: {}
    mean: {}
    std:  {}""".format(SEED, RAND_MEAN, RAND_STD))

    rand.setup(SEED)
