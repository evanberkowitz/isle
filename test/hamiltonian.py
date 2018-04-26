#!/usr/bin/env python

"""
Unittest for Hamiltonian.
"""

import unittest     # unittest module

import numpy as np

import core                 # base setup and import
core.prepare_module_import()
import cns                  # C++ bindings
import rand                 # random initializer

# RNG params
SEED = 8613
RAND_MEAN = 0
RAND_STD = 0.2
N_REP = 10 # number of repetitions


class _DummyAction(cns.Action):
    def eval(self, phi):
        return np.sum(phi)

    def force(self, phi):
        return phi

def _randomPhi(n):
    "Return a normally distributed random complex vector of n elements."
    real = np.random.normal(RAND_MEAN, RAND_STD, n)
    imag = np.random.normal(RAND_MEAN, RAND_STD, n)
    return cns.Vector(real + 1j*imag)


def _testConstructionPureCNXX():
    logger = core.get_logger()
    logger.info("Testing construction of Hamiltonian from cnxx actions")

    # anonymous variable, init method
    ham = cns.Hamiltonian(cns.HubbardGaugeAction(1))
    del ham

    # named variable
    act = cns.HubbardGaugeAction(1)
    ham = cns.Hamiltonian(act)
    del act
    del ham

    # via add method
    ham = cns.Hamiltonian()
    ham.add(cns.HubbardGaugeAction(1))
    del ham

    # via both methods
    act = cns.HubbardGaugeAction(1)
    ham = cns.Hamiltonian(cns.HubbardGaugeAction(2))
    ham.add(act)
    del ham
    del act

def _testConstructionPurePy():
    logger = core.get_logger()
    logger.info("Testing construction of Hamiltonian from Python actions")

    # anonymous variable, init method
    ham = cns.Hamiltonian(_DummyAction())
    del ham

    # named variable
    act = _DummyAction()
    ham = cns.Hamiltonian(act)
    del act
    del ham

    # via add method
    ham = cns.Hamiltonian()
    ham.add(_DummyAction())
    del ham

    # via both methods
    act = _DummyAction()
    ham = cns.Hamiltonian(_DummyAction())
    ham.add(act)
    del ham
    del act

def _testConstructionMixed():
    logger = core.get_logger()
    logger.info("Testing construction of Hamiltonian from cnxx and Python actions")

    # anonymous variable, init method
    ham = cns.Hamiltonian(_DummyAction(), cns.HubbardGaugeAction(1))
    del ham

    # named variable
    act = _DummyAction()
    ham = cns.Hamiltonian(cns.HubbardGaugeAction(1), act)
    del act
    del ham

    # via add method
    ham = cns.Hamiltonian()
    ham.add(_DummyAction())
    ham.add(cns.HubbardGaugeAction(1))
    del ham

    # via both methods
    act = _DummyAction()
    ham = cns.Hamiltonian(_DummyAction())
    ham.add(act)
    del act
    ham.add(cns.HubbardGaugeAction(1))
    act = cns.HubbardGaugeAction(1)
    ham.add(act)
    del ham
    del act


class TestHamiltonian(unittest.TestCase):
    def test_1_construction(self):
        "Test construction (and tear down) of cns.Hamiltonian."
        _testConstructionPureCNXX()
        _testConstructionPurePy()
        _testConstructionMixed()

    def test_2_eval(self):
        "Test eval function of cns.Hamiltonian."

        logger = core.get_logger()
        logger.info("Testing Hamiltonian.eval()")
        for rep in range(N_REP):
            ham = cns.Hamiltonian(_DummyAction(), cns.HubbardGaugeAction(1))
            phi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                  + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)
            pi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                 + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)

            hameval = ham.eval(cns.Vector(phi), cns.Vector(pi))
            manualeval = np.linalg.norm(pi)**2/2 + np.sum(phi) + np.linalg.norm(phi)**2/2/1

            self.assertAlmostEqual(hameval, manualeval, places=10,
                                   msg="Failed check of Hamiltonian.eval in repetition {}\n".format(rep)\
                                   + f"with hameval = {hameval}, manualeval = {manualeval}")
    def test_3_force(self):
        "Test force function of cns.Hamiltonian."

        logger = core.get_logger()
        logger.info("Testing Hamiltonian.force()")
        for rep in range(N_REP):
            ham = cns.Hamiltonian(_DummyAction(), cns.HubbardGaugeAction(1))
            phi = np.random.normal(RAND_MEAN, RAND_STD, 1000) \
                  + 1j*np.random.normal(RAND_MEAN, RAND_STD, 1000)

            hamforce = ham.force(cns.Vector(phi))
            manualforce = phi - phi/1

            self.assertAlmostEqual(np.linalg.norm(hamforce-manualforce), 0., places=10,
                                   msg="Failed check of Hamiltonian.force in repetition {}\n".format(rep)\
                                   + f"with hamforce = {hamforce}, manualforce = {manualforce}")

def setUpModule():
    "Setup the vector test module."

    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    mean: {}
    std:  {}""".format(SEED, RAND_MEAN, RAND_STD))

    rand.setup(SEED)


if __name__ == "__main__":
    unittest.main()
