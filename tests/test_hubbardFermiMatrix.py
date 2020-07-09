r"""!
Unittest for HubbardFermiMatrix.
"""

import unittest
import itertools

import numpy as np

import isle
from . import core
from . import rand

# RNG params
SEED = 8613
RAND_MEAN = 0
RAND_STD = 0.2
N_REP = 1 # number of repetitions

#  test on these lattices
LATTICES = [isle.LATTICES[name] for name in
            ("c20",
             "tube_3-3_1",
             "one_site",
             "two_sites",
             "triangle")]

# test with these values of chemical potential
MU = (0, 1, 1.5)

# types of hubbard fermi matrices
HFMS = (isle.HubbardFermiMatrixDia, isle.HubbardFermiMatrixExp)


def _randomPhi(n, real=True, imag=True):
    "Return a normally distributed random complex vector of n elements."
    r = np.random.normal(RAND_MEAN, RAND_STD, n) if real else 0
    i = np.random.normal(RAND_MEAN, RAND_STD, n) if imag else 0
    return isle.Vector(r + 1j*i)


class TestHubbardFermiMatrix(unittest.TestCase):
    def _testConstructionNt1(self, HFM, kappa, mu, sigmaKappa):
        "Check if nt=1 HFM is constructed properly."

        nt = 1
        nx = kappa.rows()
        hfm = HFM(kappa, mu, sigmaKappa)
        phi = _randomPhi(nx*nt)

        auto = np.array(isle.Matrix(hfm.Q(phi)), copy=False)
        manual = np.array(isle.Matrix(hfm.P()+hfm.Tplus(0, phi)+hfm.Tminus(0, phi)), copy=False)
        self.assertTrue(core.isEqual(auto, manual),
                        msg=f"Failed equality check for construction of {HFM} "\
                        + "for nt={}, mu={}, sigmaKappa={}".format(nt, mu, sigmaKappa)\
                        + "\nhfm.Q() = {}".format(auto) \
                        + "\nhfm.P() + hfm.Tplus(0) + hfm.Tminus(0) = {}".format(manual))

    def _testConstructionNt2(self, HFM, kappa, mu, sigmaKappa):
        "Check if nt=2 HFM is constructed properly."

        nt = 2
        nx = kappa.rows()
        hfm = HFM(kappa, mu, sigmaKappa)
        phi = _randomPhi(nx*nt)

        auto = np.array(isle.Matrix(hfm.Q(phi)), copy=False) # full matrix
        manual = np.empty(auto.shape, auto.dtype)
        P = np.array(isle.Matrix(hfm.P())) # diagonal blocks
        manual[:nx, :nx] = P
        manual[nx:, nx:] = P
        # upper off-diagonal
        manual[:nx, nx:] = isle.Matrix(hfm.Tminus(0, phi) + hfm.Tplus(0, phi))
        # lower off-diagonal
        manual[nx:, :nx] = isle.Matrix(hfm.Tminus(1, phi) + hfm.Tplus(1, phi))

        self.assertTrue(core.isEqual(auto, manual),
                        msg=f"Failed equality check for construction of {HFM} "\
                        + "for nt={}, mu={}, sigmaKappa={}".format(nt, mu, sigmaKappa)\
                        + "\nauto = {}".format(auto) \
                        + "\nmanual {}".format(manual))

    def _testConstructionNt3(self, HFM, kappa, mu, sigmaKappa):
        "Check if nt=3 HFM is constructed properly."

        nt = 3
        nx = kappa.rows()
        hfm = HFM(kappa, mu, sigmaKappa)
        phi = _randomPhi(nx*nt)

        auto = np.array(isle.Matrix(hfm.Q(phi)), copy=False) # full matrix
        manual = np.empty(auto.shape, auto.dtype)
        P = np.array(isle.Matrix(hfm.P())) # diagonal blocks

        manual[:nx, :nx] = P
        manual[:nx, nx:2*nx] = isle.Matrix(hfm.Tminus(0, phi))
        manual[:nx, 2*nx:] = isle.Matrix(hfm.Tplus(0, phi))

        manual[nx:2*nx, :nx] = isle.Matrix(hfm.Tplus(1, phi))
        manual[nx:2*nx, nx:2*nx] = P
        manual[nx:2*nx, 2*nx:] = isle.Matrix(hfm.Tminus(1, phi))

        manual[2*nx:, :nx] = isle.Matrix(hfm.Tminus(2, phi))
        manual[2*nx:, nx:2*nx] = isle.Matrix(hfm.Tplus(2, phi))
        manual[2*nx:, 2*nx:] = P

        self.assertTrue(core.isEqual(auto, manual),
                        msg="Failed equality check for construction of {HFM} "\
                        + "for nt={}, mu={}, sigmaKappa={}".format(nt, mu, sigmaKappa)\
                        + "\nauto = {}".format(auto) \
                        + "\nmanual {}".format(manual))

    def _testConstruction(self, kappa):
        "Check if HFM is constructed correctly for several values of mu and nt."

        for mu, sigmaKappa, _ in itertools.product(MU, (-1, 1),
                                                   range(N_REP)):
            for HFM in HFMS:
                self._testConstructionNt1(HFM, kappa, mu, sigmaKappa)
                self._testConstructionNt2(HFM, kappa/2, mu/2, sigmaKappa)
                self._testConstructionNt3(HFM, kappa/3, mu/3, sigmaKappa)

    def test_1_construction(self):
        "Test construction of sparse matrix for different lattices and parameters."

        logger = core.get_logger()
        for lattice in LATTICES:
            logger.info("Testing constructor of HubbardFermiMatrix* for lattice %s", lattice.name)
            self._testConstruction(lattice.hopping())

    def _test_logdetM(self, HFM, kappa):
        "Test log(det(M))."

        nx = kappa.rows()
        self.assertRaises(RuntimeError,
                          lambda msg:
                          isle.logdetM(HFM(kappa, 1, 1), isle.CDVector(nx), isle.Species.PARTICLE),
                          msg="logdetM must throw a RuntimeError when called with mu != 0. If this bug has been fixed, update the unit test!")

        for nt, beta,  mu, sigmaKappa, species, rep in itertools.product((4, 8, 32),
                                                                         (3, 6),
                                                                         [0],
                                                                         (-1, 1),
                                                                         (isle.Species.PARTICLE, isle.Species.HOLE),
                                                                         range(N_REP)):
            hfm = HFM(kappa*beta/nt, mu*beta/nt, sigmaKappa)

            phi = _randomPhi(nx*nt, imag=False)
            plain = isle.logdet(isle.Matrix(hfm.M(phi, species)))
            viaLU = isle.logdetM(hfm, phi, species)
            self.assertAlmostEqual((plain-viaLU)/max(abs(plain), abs(viaLU)), 0, places=5,
                                   msg="Failed check log(det(M)) in repetition {}".format(rep)\
                                   + "for nt={}, beta={}, mu={}, sigmaKappa={}, species={}:".format(nt, beta, mu, sigmaKappa, species)\
                                   + "\nplain = {}".format(plain) \
                                   + "\nviaLU = {}".format(viaLU))

            phi = _randomPhi(nx*nt, real=False)
            plain = isle.logdet(isle.Matrix(hfm.M(phi, species)))
            viaLU = isle.logdetM(hfm, phi, species)
            self.assertAlmostEqual((plain-viaLU)/max(abs(plain), abs(viaLU)), 0, places=5,
                                   msg="Failed check log(det(M)) in repetition {}".format(rep)\
                                   + "for nt={}, beta={}, mu={}, sigmaKappa={}, species={}:".format(nt, beta, mu, sigmaKappa, species)\
                                   + "\nplain = {}".format(plain) \
                                   + "\nviaLU = {}".format(viaLU))

            phi = _randomPhi(nx*nt)
            plain = isle.logdet(isle.Matrix(hfm.M(phi, species)))
            viaLU = isle.logdetM(hfm, phi, species)
            self.assertAlmostEqual((plain-viaLU)/max(abs(plain), abs(viaLU)), 0, places=5,
                                   msg="Failed check log(det(M)) in repetition {}".format(rep)\
                                   + "for nt={}, beta={}, mu={}, sigmaKappa={}, species={}:".format(nt, beta, mu, sigmaKappa, species)\
                                   + "\nplain = {}".format(plain) \
                                   + "\nviaLU = {}".format(viaLU))

    def _test_logdetQ(self, HFM, kappa):
        "Test log(det(Q))."

        nx = kappa.rows()
        for nt, mu, sigmaKappa, rep in itertools.product((4, 8, 32), [0],
                                                         (-1, 1), range(N_REP)):
            hfm = HFM(kappa/nt, mu/nt, sigmaKappa)
            phi = _randomPhi(nx*nt)

            plain = isle.logdet(isle.Matrix(hfm.Q(phi)))
            viaLU = isle.logdetQ(hfm, phi)

            self.assertAlmostEqual(plain, viaLU, places=10,
                                   msg="Failed check log(det(Q)) in repetition {}".format(rep)\
                                   + "for nt={}, mu={}, sigmaKappa={}:".format(nt, mu, sigmaKappa)\
                                   + "\nplain = {}".format(plain) \
                                   + "\nviaLU = {}".format(viaLU))

    def test_2_logdet(self):
        "Test log(det(M)) and log(deg(Q))."
        logger = core.get_logger()
        for lattice in LATTICES:
            logger.info("Testing log(det(M)) and log(det(Q)) %s", lattice.name)
            for HFM in HFMS:
                self._test_logdetM(HFM, lattice.hopping())
                self._test_logdetQ(HFM, lattice.hopping())


    def _test_solveM(self, HFM, kappa):
        "Test solveM()."

        nx = kappa.rows()
        for nt, mu, sigmaKappa, species, rep in itertools.product((4, 8, 32),
                                                                  [0],
                                                                  (-1, 1),
                                                                  (isle.Species.PARTICLE, isle.Species.HOLE),
                                                                  range(N_REP)):
            hfm = HFM(kappa/nt, mu/nt, sigmaKappa)
            phi = _randomPhi(nx*nt)
            M = hfm.M(phi, species)
            rhss = np.array([_randomPhi(nx*nt) for _ in range(2)])

            res = np.array(isle.solveM(hfm, phi, species, rhss), copy=False)
            chks = [np.max(np.abs(M*r-rhs)) for r, rhs in zip(res, rhss)]
            for chk in chks:
                self.assertAlmostEqual(chk, 0, places=10,
                                       msg="Failed check solveM in repetition {}".format(rep)\
                                       + "for nt={}, mu={}, sigmaKappa={}, species={}:".format(nt, mu, sigmaKappa, species)\
                                       + "\nMx - rhs = {}".format(chk))

    def test_3_solver(self):
        "Test Ax=b solvers."
        logger = core.get_logger()
        for lattice in LATTICES:
            logger.info("Testing solveM on %s", lattice.name)
            for HFM in HFMS:
                self._test_solveM(HFM, lattice.hopping())


def setUpModule():
    "Setup the HFM test module."

    logger = core.get_logger()
    logger.info("""Parameters for RNG:
    seed: {}
    mean: {}
    std:  {}""".format(SEED, RAND_MEAN, RAND_STD))

    rand.setup(SEED)
