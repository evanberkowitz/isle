#!/usr/bin/env python

"""
Unittest for Hamiltonian.
"""

import unittest

import numpy as np

import isle
from . import core
from . import rand


# RNG params
SEED = 8613
RAND_MEAN = 0
RAND_STD = 0.2
N_REP = 10 # number of repetitions


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
    logger = core.get_logger()
    logger.info("Testing construction of Hamiltonian from C++ actions")

    # anonymous variable, init method
    ham = isle.Hamiltonian(isle.action.HubbardGaugeAction(1))
    del ham

    # named variable
    act = isle.action.HubbardGaugeAction(1)
    ham = isle.Hamiltonian(act)
    del act
    del ham

    # via add method
    ham = isle.Hamiltonian()
    ham.add(isle.action.HubbardGaugeAction(1))
    del ham

    # via both methods
    act = isle.action.HubbardGaugeAction(1)
    ham = isle.Hamiltonian(isle.action.HubbardGaugeAction(2))
    ham.add(act)
    del ham
    del act

def _testConstructionPurePy():
    logger = core.get_logger()
    logger.info("Testing construction of Hamiltonian from Python actions")

    # anonymous variable, init method
    ham = isle.Hamiltonian(_DummyAction())
    del ham

    # named variable
    act = _DummyAction()
    ham = isle.Hamiltonian(act)
    del act
    del ham

    # via add method
    ham = isle.Hamiltonian()
    ham.add(_DummyAction())
    del ham

    # via both methods
    act = _DummyAction()
    ham = isle.Hamiltonian(_DummyAction())
    ham.add(act)
    del ham
    del act

def _testConstructionMixed():
    logger = core.get_logger()
    logger.info("Testing construction of Hamiltonian from C++ and Python actions")

    # anonymous variable, init method
    ham = isle.Hamiltonian(_DummyAction(), isle.action.HubbardGaugeAction(1))
    del ham

    # named variable
    act = _DummyAction()
    ham = isle.Hamiltonian(isle.action.HubbardGaugeAction(1), act)
    del act
    del ham

    # via add method
    ham = isle.Hamiltonian()
    ham.add(_DummyAction())
    ham.add(isle.action.HubbardGaugeAction(1))
    del ham

    # via both methods
    act = _DummyAction()
    ham = isle.Hamiltonian(_DummyAction())
    ham.add(act)
    del act
    ham.add(isle.action.HubbardGaugeAction(1))
    act = isle.action.HubbardGaugeAction(1)
    ham.add(act)
    del ham
    del act


class TestHamiltonian(unittest.TestCase):
    def test_1_construction(self):
        "Test construction (and tear down) of isle.Hamiltonian."
        _testConstructionPureCXX()
        _testConstructionPurePy()
        _testConstructionMixed()

    def test_2_eval(self):
        "Test eval function of isle.Hamiltonian."

        logger = core.get_logger()
        logger.info("Testing Hamiltonian.eval()")
        for rep in range(N_REP):
            ham = isle.Hamiltonian(_DummyAction(), isle.action.HubbardGaugeAction(1))
            phi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                  + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)
            pi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                 + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)

            hameval = ham.eval(isle.Vector(phi), isle.Vector(pi))
            manualeval = np.linalg.norm(pi)**2/2 + np.sum(phi) + np.linalg.norm(phi)**2/2/1

            self.assertAlmostEqual(hameval, manualeval, places=10,
                                   msg="Failed check of Hamiltonian.eval in repetition {}\n".format(rep)\
                                   + f"with hameval = {hameval}, manualeval = {manualeval}")
    def test_3_force(self):
        "Test force function of isle.Hamiltonian."

        logger = core.get_logger()
        logger.info("Testing Hamiltonian.force()")
        for rep in range(N_REP):
            ham = isle.Hamiltonian(_DummyAction(),
                                   isle.action.HubbardGaugeAction(1))
            phi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                  + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)

            hamforce = ham.force(isle.Vector(phi))
            manualforce = phi - phi/1

            self.assertAlmostEqual(np.linalg.norm(hamforce-manualforce), 0., places=10,
                                   msg="Failed check of Hamiltonian.force in repetition {}\n".format(rep)\
                                   + f"with hamforce = {hamforce}, manualforce = {manualforce}")

def setUpModule():
    "Setup the Hamiltonian test module."

    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    mean: {}
    std:  {}""".format(SEED, RAND_MEAN, RAND_STD))

    rand.setup(SEED)
