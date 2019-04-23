r"""!\file
\ingroup evolvers

"""

from copy import deepcopy
from math import sqrt, exp, floor, ceil
from logging import getLogger
from itertools import chain

import numpy as np
from scipy.stats import norm, skewnorm
from scipy.optimize import least_squares, curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .evolver import Evolver
from .selector import BinarySelector
from .. import leapfrog


TARGET_ACC_RATE = 0.67

MAX_NSTEP = 1000
ARTIFICIAL_POINTS = [(0, 0.0, 1e-8), (MAX_NSTEP, 1.0, 1e-8)]

# targeted confidence for acc rate and probability per nMD
TARGET_CONF = 0.92
TARGET_CONF_INT_ACC = 0.25 / 10
TARGET_CONF_INT_PROB = 0.25

ONE_SIGMA_PROB = 0.682689492
TWO_SIGMA_PROB = 0.954499736


def _confIntTrajPoints(trajPoints, quantileProb):
    '''
    Confidence Interval
    The formula is from paper:
    Vollset, S.E., 1993. Confidence intervals for a binomial proportion. Statistics in medicine, 12(9), pp.809-824.

    More information:
    Newcombe, Robert G. "Two-Sided Confidence Intervals for the Single Proportion: Comparison of Seven Methods," Statistics in Medicine, 17, 857-872 (1998).
    http://vassarstats.net/prop1.html
    https://www.medcalc.org/manual/values_of_the_normal_distribution.php
    http://slideplayer.com/slide/5055000/

    Assumes the probability to accept follows a normal distribution.
    Uses binomial confidence interval because we can either accept or reject, nothing more (yet).
    '''

    accepted = trajPoints.count(1)
    total = len(trajPoints)

    # quantile such that norm.cdf(quantile) == quantile_prob
    # gives the x value for the boundary of the quantileProb interval (e.g. 92% interval)
    quantile = norm.ppf(quantileProb)

    # acceptance rate
    mean = accepted / total
    # the wald interval relative to mean
    interval = quantile/sqrt(total) * sqrt((accepted/total)*(1-accepted/total)) + 1/(2*total)

    # TODO is it valid to just take the max/min? Does the CDF need to be modified for that?
    # # used with binary selector => can't be < 0
    # lower = max(0, mean - interval)
    # # used with binary selector => can't be > 1
    # higher = min(1, mean + interval)

    lower = mean - interval
    upper = mean + interval

    return lower, upper


def _errorTrajPoints(trajPoints, quantileProb):
    # TODO is this valid??
    return _intervalLength(_confIntTrajPoints(trajPoints, quantileProb)) \
        / 2 / sqrt(len(trajPoints))


def _confIntProbabilities(probabilities, quantileProb):
    mean = np.mean(probabilities)
    err = np.std(probabilities) / sqrt(len(probabilities))
    # endpoints of quantileProb confidence interval
    result = norm.interval(quantileProb, loc=mean, scale=err)
    # TODO see above
    # # used with probabilities => can't be < 0
    # lower = max(0, result[0])
    # # used with probabilities => can't be > 1
    # higher = min(1, result[1])

    lower = result[0]
    upper = result[1]

    return lower, upper

def _errorProbabilities(probabilities, quantileProb):
    # divide by two because standard deviation is only half the confidence interval
    return _intervalLength(_confIntProbabilities(probabilities, quantileProb)) \
        / 2 / sqrt(len(probabilities))

def _intervalLength(interval):
    return interval[1] - interval[0]


class Recording:

    class Entry:
        def __init__(self, length, nstep):
            self.length = length
            self.nstep = nstep
            self.probabilities = []
            self.trajPoints = []

        def __len__(self):
            return len(self.trajPoints)

        def add(self, probability, trajPoint):
            self.probabilities.append(probability)
            self.trajPoints.append(trajPoint)

        def confIntProbabilities(self, quantileProb):
            return _confIntProbabilities(self.probabilities, quantileProb)

        def confIntTrajPoints(self, quantileProb):
            return _confIntTrajPoints(self.trajPoints, quantileProb)


    def __init__(self, initialLength, initialNstep):
        self.entries = []
        self._paramToEntry = dict()

        self.newEntry(initialLength, initialNstep)

    def __len__(self):
        return len(self.entries)

    def current(self):
        return self.entries[-1]

    def __getitem__(self, index):
        return self.entries[index]

    def newEntry(self, length, nstep):
        entry = self.Entry(length, nstep)
        self.entries.append(entry)

        params = (length, nstep)
        try:
            self._paramToEntry[params].append(entry)
        except KeyError:
            self._paramToEntry[params] = [entry]

        return entry

    def gather(self, length=None, nstep=None):
        if length is None:
            if nstep is None:
                raise ValueError("One of length and nstep must not be None")
            # filter with respect to nstep
            paramFilter = lambda keyVal: keyVal[0][1] == nstep
            # and use length as key
            selectParam = lambda params: params[0]
        else:
            if nstep is not None:
                raise ValueError("One of length and nstep must be None")
            # filter with respect to length
            paramFilter = lambda keyVal: keyVal[0][0] == length
            # and use nstep as key
            selectParam = lambda params: params[1]

        probPoints = []
        tpPoints = []
        for params, entries in filter(paramFilter, self._paramToEntry.items()):
            # flatten arrays
            probs = [prob for entry in entries for prob in entry.probabilities]
            tps = [tp for entry in entries for tp in entry.probabilities]

            # store varying parameter, mean, and error
            probPoints.append((selectParam(params), np.mean(probs),
                               _errorProbabilities(probs, ONE_SIGMA_PROB)))
            tpPoints.append((selectParam(params), np.mean(tps),
                             _errorTrajPoints(tps, ONE_SIGMA_PROB)))

        return probPoints, tpPoints


# def fitFunctionLS(a, x, y):
#     return skewnorm.cdf(x, *a) - y

# def fitFunctionCF(x, *a):
#     return skewnorm.cdf(x, *a)

# def fitData(runData):
#     points = runData.probAsPoints() + runData.tpAsPoints() + ARTIFICIAL_POINTS
#     indep, dep, deperr = np.asarray(points).T
#     deperr[deperr == 0.0] = 1e-8
#     return indep, dep, deperr


# def squareSum(func, indep, dep, deperr, par):
#     return np.sum((func(indep, *par)-dep)**2 / deperr**2)


# class Fitter:
#     class Result:
#         def __init__(self, bestFit, otherFits):
#             self.bestFit = bestFit
#             self.otherFits = otherFits
#             self.optimum = skewnorm.ppf(TARGET_ACC_RATE, *bestFit[0])

#         def plot(self, ax, xvalues, color="k", ls="-"):
#             ax.plot(xvalues, fitFunctionCF(xvalues, *self.bestFit[0]), c=color, ls=ls)
#             for other in self.otherFits:
#                 ax.plot(xvalues, fitFunctionCF(xvalues, *other[0]), c=color, ls=ls, alpha=0.5)

#             ax.axvline(self.optimum, c=color, ls="--")

#     def __init__(self):
#         self._startParams = [(0.1, 1, 10),
#                              (1, 1, 1)]
#         self._lastFit = None


#     def fit(self, runData):
#         independent, dependent, dependenterr = fitData(runData)

#         startParams = self._startParams + (self._lastFit if self._lastFit is not None else [])
#         # # w/o errors
#         # result = min((least_squares(fitFunctionLS, guess, max_nfev=1000,
#         #                             args=(independent, dependent),
#         #                             verbose=0, loss="soft_l1", method="trf")
#         #               for guess in startParams),
#         #              key=lambda res: res.cost)

#         # w/ errors
#         results = sorted([curve_fit(fitFunctionCF, independent, dependent,
#                                     p0=guess, sigma=dependenterr,
#                                     absolute_sigma=False, method="trf")
#                           for guess in startParams],
#                          key=lambda pair: squareSum(fitFunctionCF, independent,
#                                                     dependent, dependenterr, pair[0]))
#         bestFit = results[0]
#         otherFits = results[1:]

#         # getLogger("fit").error(result)

#         self._lastFit = bestFit[0]

#         return self.Result(bestFit, otherFits)



# class LeapfrogParam:
#     def __init__(self, action, length, nstep):
#         self.action = action
#         self.length = length
#         self.nstep = nstep


# class LeapfrogTuner(Evolver):
#     r"""! \ingroup evolvers

#     """

#     def __init__(self, action, length, nstep, rng):
#         r"""!

#         """

#         self._lfParam = LeapfrogParam(action, length, nstep)
#         self._runData = RunData()
#         self._fitter = Fitter()
#         self._trace = []
#         self._selector = BinarySelector(rng)
#         self._finished = False

#     def evolve(self, phi, pi, actVal, trajPoint):
#         r"""!
#         Run leapfrog integrator.
#         \param phi Input configuration.
#         \param pi Input Momentum.
#         \param actVal Value of the action at phi.
#         \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
#         \returns In order:
#           - New configuration
#           - New momentum
#           - Action evaluated at new configuration
#           - Point along trajectory that was selected
#         """

#         phi, pi, actVal, trajPoint = self._doEvolve(phi, pi, actVal, trajPoint)

#         log = getLogger("atune")

#         if len(self._runData.probPerNStep[self._lfParam.nstep]) > 5:
#             confIntProb = self._runData.confIntProb(self._lfParam.nstep)
#             confIntTP = self._runData.confIntTP(self._lfParam.nstep)

#             # log.error(f"Prob: {confIntProb}, TP: {confIntTP}")

#             if intDiff(confIntTP) < TARGET_CONF_INT_ACC:
#                 log.error("Stopping because of tp")
#                 self._pickNextNStep()

#             if intDiff(confIntProb) < TARGET_CONF_INT_PROB:
#                 log.error("Stopping because of prob")
#                 self._pickNextNStep()

#         if self._runData.currentCount > 100:
#             raise RuntimeError("Did not converge")

#         if len(self._runData.tpPerNStep) > 10:
#             self._finished = True

#         if self._finished:
#             raise StopIteration()

#         return phi, pi, actVal, trajPoint

#     def plot(self):
#         trace = self._trace[:min(16, len(self._trace))]

#         fig = plt.figure(figsize=(11, 11))
#         gspec = gridspec.GridSpec(1, 2)
#         innerGspec = gridspec.GridSpecFromSubplotSpec(4, int(ceil(len(trace)/4)),
#                                                       subplot_spec=gspec[0, 0])

#         axs = [fig.add_subplot(innerGspec[i, j])
#                for i in range(innerGspec.get_geometry()[0])
#                for j in range(innerGspec.get_geometry()[1])]

#         for runNo, (ax, (fitResult, runData)) in enumerate(zip(axs, trace)):
#             nsteps = np.linspace(0, max(runData.tpPerNStep.keys())*1.05, 100)
#             runData.plot(ax)
#             fitResult.plot(ax, nsteps)
#             ax.set_title(f"Run {runNo}")
#             ax.legend()


#         ax = fig.add_subplot(gspec[0, 1])


#         fig.tight_layout()


#     def _pickNextNStep(self):
#         fitResult = self._fitter.fit(self._runData)
#         self._trace.append((fitResult, deepcopy(self._runData)))
#         floatStep = fitResult.optimum

#         nextStep = max(int(floor(floatStep)), 1)
#         if nextStep in self._runData.tpPerNStep:
#             nextStep = int(ceil(floatStep))
#             if nextStep in self._runData.tpPerNStep:
#                 getLogger("atune").error(f"Done with nstep = {nextStep}")
#                 self._finished = True

#         if nextStep > MAX_NSTEP:
#             raise RuntimeError(f"nstep is too large: {nextStep}")

#         self._lfParam.nstep = nextStep
#         self._runData.currentCount = 0
#         getLogger("atune").error("New nstep: %d", nextStep)

#     def _doEvolve(self, phi0, pi0, actVal0, _trajPoint0):
#         phi1, pi1, actVal1 = leapfrog(phi0, pi0, self._lfParam.action,
#                                       self._lfParam.length, self._lfParam.nstep)
#         energy0 = actVal0 + +np.linalg.norm(pi0)**2/2
#         energy1 = actVal1 + +np.linalg.norm(pi1)**2/2
#         trajPoint1 = self._selector.selectTrajPoint(energy0, energy1)

#         self._runData.addForNStep(self._lfParam.nstep,
#                                   min(1, exp(np.real(energy0 - energy1))),
#                                   trajPoint1)
#         self._runData.currentCount += 1

#         return (phi1, pi1, actVal1, trajPoint1) if trajPoint1 == 1 \
#             else (phi0, pi0, actVal0, trajPoint1)

#     def save(self, h5group, manager):
#         r"""!
#         Save the evolver to HDF5.
#         \param h5group HDF5 group to save to.
#         \param manager EvolverManager whose purview to save the evolver in.
#         """

#     @classmethod
#     def fromH5(cls, h5group, _manager, action, _lattice, rng):
#         r"""!
#         Construct from HDF5.
#         \param h5group HDF5 group to load parameters from.
#         \param _manager \e ignored.
#         \param action Action to use.
#         \param _lattice \e ignored.
#         \param rng Central random number generator for the run.
#         \returns A newly constructed leapfrog evolver.
#         """
#         return None

