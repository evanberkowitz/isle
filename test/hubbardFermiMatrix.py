#!/usr/bin/env python

"""
Unittest for HubbardFermiMatrix.
"""

import unittest     # unittest module
import itertools

import numpy as np
import yaml

import core                 # base setup and import
core.prepare_module_import()
import cns                  # C++ bindings
import rand                 # random initializer

# RNG params
SEED = 8613
RAND_MEAN = 0
RAND_STD = 0.2
N_REP = 1 # number of repetitions

# lattices to test matrix with
LATTICES = [core.TEST_PATH/"../lattices/c60_ipr.yml",]
            # core.TEST_PATH/"../lattices/tube_3-3_1.yml",
            # core.TEST_PATH/"../lattices/tube_3-3_5.yml",
            # core.TEST_PATH/"../lattices/tube_4-2_2.yml"]

# test with these values of chemical potential
# MU = [0, 1, 1.5]
MU = [1]

def _randomPhi(n):
    "Return a normally distributed random complex vector of n elements."
    real = np.random.normal(RAND_MEAN, RAND_STD, n)
    imag = np.random.normal(RAND_MEAN, RAND_STD, n)
    return cns.Vector(real + 1j*imag)


class TestHubbardFermiMatrix(unittest.TestCase):
    def _testConstructionNt1(self, kappa, mu, sigmaKappa):
        "Check if nt=1 HFM is constructed properly."

        nt = 1
        nx = kappa.rows()
        hfm = cns.HubbardFermiMatrix(kappa, _randomPhi(nx*nt),
                                     mu, sigmaKappa)

        auto = np.array(cns.Matrix(hfm.Q()), copy=False)
        manual = np.array(cns.Matrix(hfm.P()+hfm.Tplus(0)+hfm.Tminus(0)), copy=False)
        self.assertTrue(core.isEqual(auto, manual),
                        msg="Failed equality check for construction of hubbardFermiMatrix "\
                        + "for nt={}, mu={}, sigmaKappa={}".format(nt, mu, sigmaKappa)\
                        + "\nhfm.Q() = {}".format(auto) \
                        + "\nhfm.P() + hfm.Tplus(0) + hfm.Tminus(0) = {}".format(manual))

    def _testConstructionNt2(self, kappa, mu, sigmaKappa):
        "Check if nt=2 HFM is constructed properly."

        nt = 2
        nx = kappa.rows()
        hfm = cns.HubbardFermiMatrix(kappa, _randomPhi(nx*nt),
                                     mu, sigmaKappa)

        auto = np.array(cns.Matrix(hfm.Q()), copy=False) # full matrix
        manual = np.empty(auto.shape, auto.dtype)
        P = np.array(cns.Matrix(hfm.P())) # diagonal blocks
        manual[:nx, :nx] = P
        manual[nx:, nx:] = P
        manual[:nx, nx:] = cns.Matrix(hfm.Tminus(0) + hfm.Tplus(0)) # upper off diagonal
        manual[nx:, :nx] = cns.Matrix(hfm.Tminus(1) + hfm.Tplus(1)) # lower off diagonal

        self.assertTrue(core.isEqual(auto, manual),
                        msg="Failed equality check for construction of hubbardFermiMatrix "\
                        + "for nt={}, mu={}, sigmaKappa={}".format(nt, mu, sigmaKappa)\
                        + "\nauto = {}".format(auto) \
                        + "\nmanual {}".format(manual))

    def _testConstructionNt3(self, kappa, mu, sigmaKappa):
        "Check if nt=3 HFM is constructed properly."

        nt = 3
        nx = kappa.rows()
        hfm = cns.HubbardFermiMatrix(kappa, _randomPhi(nx*nt),
                                     mu, sigmaKappa)

        auto = np.array(cns.Matrix(hfm.Q()), copy=False) # full matrix
        manual = np.empty(auto.shape, auto.dtype)
        P = np.array(cns.Matrix(hfm.P())) # diagonal blocks

        manual[:nx, :nx] = P
        manual[:nx, nx:2*nx] = cns.Matrix(hfm.Tminus(0))
        manual[:nx, 2*nx:] = cns.Matrix(hfm.Tplus(0))

        manual[nx:2*nx, :nx] = cns.Matrix(hfm.Tplus(1))
        manual[nx:2*nx, nx:2*nx] = P
        manual[nx:2*nx, 2*nx:] = cns.Matrix(hfm.Tminus(1))

        manual[2*nx:, :nx] = cns.Matrix(hfm.Tminus(2))
        manual[2*nx:, nx:2*nx] = cns.Matrix(hfm.Tplus(2))
        manual[2*nx:, 2*nx:] = P

        self.assertTrue(core.isEqual(auto, manual),
                        msg="Failed equality check for construction of hubbardFermiMatrix "\
                        + "for nt={}, mu={}, sigmaKappa={}".format(nt, mu, sigmaKappa)\
                        + "\nauto = {}".format(auto) \
                        + "\nmanual {}".format(manual))

    def _testConstruction(self, kappa):
        "Check if HFM is constructed correctly for several values of mu and nt."

        for mu, sigmaKappa, _ in itertools.product(MU, (-1, 1),
                                                   range(N_REP)):
            self._testConstructionNt1(kappa, mu, sigmaKappa)
            self._testConstructionNt2(kappa/2, mu/2, sigmaKappa)
            self._testConstructionNt3(kappa/3, mu/3, sigmaKappa)

    def test_1_construction(self):
        "Test construction of sparse matrix for different lattices and parameters."

        logger = core.get_logger()
        for latfile in LATTICES:
            logger.info("Testing constructor of HubbardFermiMatrix for lattice %s", latfile)
            with open(latfile, "r") as f:
                lat = yaml.safe_load(f)
                self._testConstruction(lat.hopping())


    def _testQLUFact(self, kappa):
        "Check whether a matrix reconstructed from an LU decomposition of Q is equal to the original."
        for nt, mu, sigmaKappa, _ in itertools.product((4, 8, 32), MU, (-1, 1),
                                                       range(N_REP)):
            nx = kappa.rows()
            hfm = cns.HubbardFermiMatrix(kappa/nt, _randomPhi(nx*nt),
                                         mu/nt, sigmaKappa)
            q = np.array(cns.Matrix(hfm.Q()), copy=False)
            lu = cns.getQLU(hfm)
            recon = np.array(cns.Matrix(lu.reconstruct()))

            self.assertTrue(core.isEqual(q, recon, nOps=nx**2, prec=1e-13),
                            msg="Failed equality check of reconstruction from LU decomposition of hubbardFermiMatrix "\
                            + "for nt={}, mu={}, sigmaKappa={}".format(nt, mu, sigmaKappa)\
                            + "\noriginal = {}".format(q) \
                            + "\nreconstructed {}".format(recon))

    def test_2_lu_factorization(self):
        "Test LU factorization."
        logger = core.get_logger()
        for latfile in LATTICES:
            logger.info("Testing LU factorization of HubbardFermiMatrix for lattice %s", latfile)
            with open(latfile, "r") as f:
                lat = yaml.safe_load(f)
                self._testQLUFact(lat.hopping())


    def _test_logdetM(self, kappa):
        "Test log(det(M))."

        nx = kappa.rows()
        self.assertRaises(RuntimeError,
                          lambda msg:
                          cns.logdetM(cns.HubbardFermiMatrix(kappa, cns.CDVector(nx), 1, 1), True),
                          msg="logdetM must throw a RuntimeError when called with mu != 0. If this bug has been fixed, update the unit test!")

        for nt, mu, sigmaKappa, hole, rep in itertools.product((4, 8, 32), [0], (-1, 1),
                                                               (False, True), range(N_REP)):
            hfm = cns.HubbardFermiMatrix(kappa/nt, _randomPhi(nx*nt),
                                         mu/nt, sigmaKappa)

            plain = cns.logdet(cns.Matrix(hfm.M(hole)))
            viaLU = cns.logdetM(hfm, hole)

            self.assertAlmostEqual(plain, viaLU, places=10,
                                   msg="Failed check log(det(M)) in repetition {}".format(rep)\
                                   + "for nt={}, mu={}, sigmaKappa={}, hole={}:".format(nt, mu, sigmaKappa, hole)\
                                   + "\nplain = {}".format(plain) \
                                   + "\nviaLU = {}".format(viaLU))


    def test_3_logdet(self):
        "Test log(det(M)) and log(deg(Q))."
        logger = core.get_logger()
        for latfile in LATTICES:
            logger.info("Testing log(det(M)) and log(det(Q)) %s", latfile)
            with open(latfile, "r") as f:
                lat = yaml.safe_load(f)
                self._test_logdetM(lat.hopping())



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
