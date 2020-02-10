r"""!
Unittest for HubbardFermiAction.
"""

import unittest
from itertools import product

import numpy as np

import isle
from . import core
from . import rand

# RNG params
SEED = 8613
RAND_MEAN = 0
RAND_STD = 0.2
N_REP = 5 # number of repetitions

# lattices to test matrix with
LATTICES = [isle.LATTICES[name] for name in ("tube_3-3_1",
                                             "one_site",
                                             "two_sites",
                                             "ribbon_3agnr7",
                                             "triangle")]

# test with these values for parameters
NT = (8, 16)
MU = (0, )
BETA = (1, 4)

def _randomPhi(n, realOnly):
    "Return a normally distributed random complex vector of n elements."
    real = np.random.normal(RAND_MEAN, RAND_STD, n)
    if not realOnly:
        imag = np.random.normal(RAND_MEAN, RAND_STD, n)
    else:
        imag = 0
    return isle.Vector(real + 1j*imag)

def _forAllParams():
    yield from product((isle.action.HFAHopping.DIA, isle.action.HFAHopping.EXP),
                       (isle.action.HFABasis.PARTICLE_HOLE, isle.action.HFABasis.SPIN),
                       NT,
                       BETA,
                       MU,
                       (-1, +1))


class TestHubbardFermiAction(unittest.TestCase):

    def _testVariantsEval(self, lat, hopping, basis, beta, mu, sigmaKappa):
        actv1 = isle.action.makeHubbardFermiAction(lat,
                                                   beta,
                                                   mu*beta/lat.nt(),
                                                   sigmaKappa,
                                                   hopping,
                                                   basis,
                                                   isle.action.HFAVariant.ONE,
                                                   False)
        actv2 = isle.action.makeHubbardFermiAction(lat,
                                                   beta,
                                                   mu*beta/lat.nt(),
                                                   sigmaKappa,
                                                   hopping,
                                                   basis,
                                                   isle.action.HFAVariant.TWO,
                                                   False)

        for rep in range(N_REP):
            phi = _randomPhi(lat.lattSize(), False)

            self.assertAlmostEqual(
                actv1.eval(phi), actv2.eval(phi), places=9,
                msg=f"Failed check of evaluation of action in repetition {rep} "\
                + f"for lat={lat.name}, nt={lat.nt()}, mu={mu}, sigmaKappa={sigmaKappa}, "\
                + f"beta={beta}, hopping={hopping}, basis={basis}")

    def _testShortcutEval(self, lat, hopping, basis, beta, mu, sigmaKappa):
        """
        Test shortcut for hole determinant.
        Only some of the cases tested here actually use the shortcut.
        """

        actnoshort = isle.action.makeHubbardFermiAction(lat,
                                                        beta,
                                                        mu*beta/lat.nt(),
                                                        sigmaKappa,
                                                        hopping,
                                                        basis,
                                                        isle.action.HFAVariant.ONE,
                                                        False)
        actshort = isle.action.makeHubbardFermiAction(lat,
                                                      beta,
                                                      mu*beta/lat.nt(),
                                                      sigmaKappa,
                                                      hopping,
                                                      basis,
                                                      isle.action.HFAVariant.ONE,
                                                      True)

        for rep in range(N_REP):
            phi = _randomPhi(lat.lattSize(), True)

            self.assertAlmostEqual(
                actnoshort.eval(phi), actshort.eval(phi), places=10,
                msg=f"Failed check of shortcut in evaluation of action in repetition {rep} "\
                + f"for lat={lat.name}, nt={lat.nt()}, mu={mu}, sigmaKappa={sigmaKappa}, "\
                + f"beta={beta}, hopping={hopping}, basis={basis}")


    def test_1_eval(self):
        "Test eval functions of all versions of the action."

        for lat in LATTICES:
            for hopping, basis, nt, beta, mu, sigmaKappa in _forAllParams():
                lat.nt(nt)
                self._testVariantsEval(lat, hopping, basis, beta, mu, sigmaKappa)
                self._testShortcutEval(lat, hopping, basis, beta, mu, sigmaKappa)


    def _testVariantsForce(self, lat, hopping, basis, beta, mu, sigmaKappa):
        actv1 = isle.action.makeHubbardFermiAction(lat,
                                                   beta,
                                                   mu*beta/lat.nt(),
                                                   sigmaKappa,
                                                   hopping,
                                                   basis,
                                                   isle.action.HFAVariant.ONE,
                                                   False)
        actv2 = isle.action.makeHubbardFermiAction(lat,
                                                   beta,
                                                   mu*beta/lat.nt(),
                                                   sigmaKappa,
                                                   hopping,
                                                   basis,
                                                   isle.action.HFAVariant.TWO,
                                                   False)

        for rep in range(N_REP):
            phi = _randomPhi(lat.lattSize(), False)

            self.assertAlmostEqual(
                np.max(np.abs(actv1.force(phi)-actv2.force(phi))), 0, places=10,
                msg=f"Failed check of force from action in repetition {rep} "\
                + f"for lat={lat.name}, nt={lat.nt()}, mu={mu}, sigmaKappa={sigmaKappa}, "\
                + f"beta={beta}, hopping={hopping}, basis={basis}")

    def _testShortcutForce(self, lat, hopping, basis, beta, mu, sigmaKappa):
        """
        Test shortcut for hole determinant.
        Only some of the cases tested here actually use the shortcut.
        """

        actnoshort = isle.action.makeHubbardFermiAction(lat,
                                                        beta,
                                                        mu*beta/lat.nt(),
                                                        sigmaKappa,
                                                        hopping,
                                                        basis,
                                                        isle.action.HFAVariant.ONE,
                                                        False)
        actshort = isle.action.makeHubbardFermiAction(lat,
                                                      beta,
                                                      mu*beta/lat.nt(),
                                                      sigmaKappa,
                                                      hopping,
                                                      basis,
                                                      isle.action.HFAVariant.ONE,
                                                      True)

        for rep in range(N_REP):
            phi = _randomPhi(lat.lattSize(), True)

            self.assertAlmostEqual(
                np.max(np.abs(actnoshort.force(phi)-actshort.force(phi))), 0, places=10,
                msg=f"Failed check of shortcut in evaluation of force in repetition {rep} "\
                + f"for lat={lat.name}, nt={lat.nt()}, mu={mu}, sigmaKappa={sigmaKappa}, "\
                + f"beta={beta}, hopping={hopping}, basis={basis}")


    def test_2_force(self):
        "Test force functions of all versions of the action."

        for lat in LATTICES:
            for hopping, basis, nt, beta, mu, sigmaKappa in _forAllParams():
                lat.nt(nt)
                self._testVariantsForce(lat, hopping, basis, beta, mu, sigmaKappa)
                self._testShortcutForce(lat, hopping, basis, beta, mu, sigmaKappa)


def setUpModule():
    "Setup the HFM test module."

    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    mean: {}
    std:  {}""".format(SEED, RAND_MEAN, RAND_STD))

    rand.setup(SEED)
