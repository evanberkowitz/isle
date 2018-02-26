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

# lattices to test matrix with
LATTICES = [core.TEST_PATH/"../lattices/c60_ipr.yml",
            core.TEST_PATH/"../lattices/tube_3-3_1.yml",
            core.TEST_PATH/"../lattices/tube_3-3_5.yml",
            core.TEST_PATH/"../lattices/tube_4-2_2.yml"]

# test with these values of chemical potential
MU = [0, 1, 1.5]

def _randomPhi(n):
    "Return a normally distributed random complex vector of n elements."
    real = np.random.normal(RAND_MEAN, RAND_STD, n)
    imag = np.random.normal(RAND_MEAN, RAND_STD, n)
    return cns.Vector(real + 1j*imag)


class TestHubbardFermiMatrix(unittest.TestCase):
    def _testConstructionNt1(self, kappa, mu, sigmaMu, sigmaKappa):
        "Check if nt=1 HFM is constructed properly."

        nt = 1
        hfm = cns.HubbardFermiMatrix(kappa, _randomPhi(nx*nt),
                                     mu, sigmaMu, sigmaKappa)

        auto = np.array(cns.Matrix(hfm.MMdag()), copy=False)
        manual = np.array(cns.Matrix(hfm.P()+hfm.Q(0)+hfm.Qdag(0)), copy=False)
        self.assertTrue(core.isEqual(auto, manual),
                        msg="Failed equality check for construction of hubbardFermiMatrix "\
                        + "for nt={}, mu={}, sigmaMu={}, sigmaKappa={}".format(nt, mu, sigmaMu, sigmaKappa)\
                        + "\nhfm.MMdag() = {}".format(auto) \
                        + "\nhfm.P() + hfm.Q(0) + hfm.Qdag(0) = {}".format(manual))

    def _testConstructionNt2(self, kappa, mu, sigmaMu, sigmaKappa):
        "Check if nt=2 HFM is constructed properly."

        nt = 2
        nx = kappa.rows()
        hfm = cns.HubbardFermiMatrix(kappa, _randomPhi(nx*nt),
                                     mu, sigmaMu, sigmaKappa)

        auto = np.array(cns.Matrix(hfm.MMdag()), copy=False) # full matrix
        manual = np.empty(auto.shape, auto.dtype)
        P = np.array(cns.Matrix(hfm.P())) # diagonal blocks
        manual[:nx, :nx] = P
        manual[nx:, nx:] = P
        manual[:nx, nx:] = cns.Matrix(hfm.Qdag(0) + hfm.Q(0)) # upper off diagonal
        manual[nx:, :nx] = cns.Matrix(hfm.Qdag(1) + hfm.Q(1)) # lower off diagonal

        self.assertTrue(core.isEqual(auto, manual),
                        msg="Failed equality check for construction of hubbardFermiMatrix "\
                        + "for nt={}, mu={}, sigmaMu={}, sigmaKappa={}".format(nt, mu, sigmaMu, sigmaKappa)\
                        + "\nauto = {}".format(auto) \
                        + "\nmanual {}".format(manual))

    def _testConstructionNt3(self, kappa, mu, sigmaMu, sigmaKappa):
        "Check if nt=3 HFM is constructed properly."

        nt = 3
        nx = kappa.rows()
        hfm = cns.HubbardFermiMatrix(kappa, _randomPhi(nx*nt),
                                     mu, sigmaMu, sigmaKappa)

        auto = np.array(cns.Matrix(hfm.MMdag()), copy=False) # full matrix
        manual = np.empty(auto.shape, auto.dtype)
        P = np.array(cns.Matrix(hfm.P())) # diagonal blocks

        manual[:nx, :nx] = P
        manual[:nx, nx:2*nx] = cns.Matrix(hfm.Qdag(0))
        manual[:nx, 2*nx:] = cns.Matrix(hfm.Q(0))

        manual[nx:2*nx, :nx] = cns.Matrix(hfm.Q(1))
        manual[nx:2*nx, nx:2*nx] = P
        manual[nx:2*nx, 2*nx:] = cns.Matrix(hfm.Qdag(1))

        manual[2*nx:, :nx] = cns.Matrix(hfm.Qdag(2))
        manual[2*nx:, nx:2*nx] = cns.Matrix(hfm.Q(2))
        manual[2*nx:, 2*nx:] = P

        self.assertTrue(core.isEqual(auto, manual),
                        msg="Failed equality check for construction of hubbardFermiMatrix "\
                        + "for nt={}, mu={}, sigmaMu={}, sigmaKappa={}".format(nt, mu, sigmaMu, sigmaKappa)\
                        + "\nauto = {}".format(auto) \
                        + "\nmanual {}".format(manual))

    def _testConstruction(self, kappa):
        "Check if HFM is constructed correctly for several values of mu and nt."

        for mu, sigmaMu, sigmaKappa in itertools.product(MU, (-1, 1), (-1, 1)):
            self._testConstructionNt1(kappa, mu, sigmaMu, sigmaKappa)
            self._testConstructionNt2(kappa, mu, sigmaMu, sigmaKappa)
            self._testConstructionNt3(kappa, mu, sigmaMu, sigmaKappa)

    def test_1_construction(self):
        "Test construction of sparse matrix for different lattices and parameters."

        logger = core.get_logger()
        for latfile in LATTICES:
            logger.info("Testing constructor of HubbardFermiMatrix for lattice %s", latfile)
            with open(latfile, "r") as f:
                lat = yaml.safe_load(f)
                self._testConstruction(lat.hopping())


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
