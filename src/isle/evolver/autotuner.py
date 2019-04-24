r"""!\file
\ingroup evolvers

"""

from copy import deepcopy
from math import sqrt, exp, floor, ceil
from logging import getLogger
from itertools import chain

import h5py as h5
import numpy as np
from scipy.stats import norm, skewnorm
from scipy.optimize import least_squares, curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .evolver import Evolver
from .selector import BinarySelector
from .. import leapfrog
from ..h5io import createH5Group, loadList


TARGET_ACC_RATE = 0.67

MAX_NSTEP = 1000


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
    interval = quantile/sqrt(total) * sqrt(mean*(1-mean)) + 1/(2*total)

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
    err = np.std(probabilities)
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


# def appendToListInDict(dictionary, key, value):
#     try:
#         dictionary[key].append(value)
#     except KeyError:
#         dictionary[key] = [value]

def extendListInDict(dictionary, key, values):
    try:
        dictionary[key].extend(values)
    except KeyError:
        dictionary[key] = deepcopy(values)


class Registrar:

    class Record:
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

        def __eq__(self, other):
            return self.length == other.length \
                and self.nstep == other.nstep \
                and self.probabilities == other.probabilities \
                and self.trajPoints == other.trajPoints \

        def __str__(self):
            return f"""Record(length={self.length}, nstep={self.nstep}
       probabilities={self.probabilities}
       trajPoints={self.trajPoints})"""

        def save(self, h5group):
            h5group["length"] = self.length
            h5group["nstep"] = self.nstep
            h5group["probabilities"] = self.probabilities
            h5group["trajPoints"] = self.trajPoints

        @classmethod
        def fromH5(cls, h5group):
            record = cls(h5group["length"][()], h5group["nstep"][()])
            record.probabilities = list(h5group["probabilities"][()])
            record.trajPoints = list(h5group["trajPoints"][()])
            return record

    def __init__(self, initialLength, initialNstep):
        self.records = []
        self._knownLength = set()
        self._knownNstep = set()
        self.newRecord(initialLength, initialNstep)

        self.fitResults = []

    def __len__(self):
        return len(self.records)

    def currentRecord(self):
        return self.records[-1]

    def newRecord(self, length, nstep):
        record = self.Record(length, nstep)
        self.records.append(record)
        self._knownLength.add(length)
        self._knownNstep.add(nstep)
        return record

    def addFitResult(self, result):
        self.fitResults.append(result)

    def gather(self, *, length=None, nstep=None, maxRecord=None):
        if length is None:
            if nstep is None:
                raise ValueError("One of length and nstep must not be None")
            # filter with respect to nstep
            recordFilter = lambda record: record.nstep == nstep
            # and use length as key
            selectParam = lambda record: record.length
        else:
            if nstep is not None:
                raise ValueError("One of length and nstep must be None")
            # filter with respect to length
            recordFilter = lambda record: record.length == length
            # and use nstep as key
            selectParam = lambda record: record.nstep

        probDict = dict()
        tpDict = dict()
        for record in filter(recordFilter, self.records[:maxRecord]):
            extendListInDict(probDict, selectParam(record), record.probabilities)
            extendListInDict(tpDict, selectParam(record), record.trajPoints)

        probabilities = [(param, np.mean(probs), _errorProbabilities(probs, ONE_SIGMA_PROB))
                         for param, probs in probDict.items()]
        trajPoints = [(param, np.mean(tps), _errorTrajPoints(tps, ONE_SIGMA_PROB))
                      for param, tps in tpDict.items()]

        return probabilities, trajPoints

    def seenBefore(self, *, length=None, nstep=None):
        if length is None:
            if nstep is None:
                raise ValueError("At leas one of length and nstep must not be None")
            return nstep in self._knownNstep

        # else: length is not None
        if nstep is None:
            return length in self._knownLength

        # else: both not None
        return length in self._knownLength and nstep in self._knownNstep

    def _saveRecords(self, h5group):
        maxStored = -1
        for idx, grp in loadList(h5group):
            storedRecord = self.Record.fromH5(grp)
            if storedRecord != self.records[idx]:
                getLogger(__name__).error("Cannot save recording, record %d stored in the file "
                                          "does dot match record in memory.", idx)
                raise RuntimeError("Record in file does not match record in memory")
            maxStored = idx

        for idx, record in filter(lambda pair: pair[0] > maxStored, enumerate(self.records)):
            record.save(h5group.create_group(str(idx)))

    def _saveFitResults(self, h5group):
        maxStored = -1
        for idx, grp in loadList(h5group):
            storedResult = Fitter.Result.fromH5(grp)
            if storedResult != self.fitResults[idx]:
                getLogger(__name__).error("Cannot save recording, fit result %d stored in the file "
                                          "does dot match fit result in memory.", idx)
                raise RuntimeError("Fit result in file does not match fit result in memory")
            maxStored = idx

        for idx, fitResult in filter(lambda pair: pair[0] > maxStored, enumerate(self.fitResults)):
            fitResult.save(h5group.create_group(str(idx)))

    def save(self, h5group):
        self._saveRecords(createH5Group(h5group, "records"))
        self._saveFitResults(createH5Group(h5group, "fitResults"))

    @classmethod
    def fromH5(cls, h5group):
        # build a completely empty instance
        registrar = cls(0, 0)
        registrar.records = []
        registrar._knownLength = set()
        registrar._knownNstep = set()

        for _, grp in sorted(h5group["records"].items(),
                                 key=lambda pair: int(pair[0])):
            storedRecord = cls.Record.fromH5(grp)
            # go through this function to make sure all internal variables are set up properly
            record = registrar.newRecord(storedRecord.length, storedRecord.nstep)
            record.probabilities = storedRecord.probabilities
            record.trajPoints = storedRecord.trajPoints

        for _, grp in sorted(h5group["fitResults"].items(),
                                 key=lambda pair: int(pair[0])):
            registrar.addFitResult(Fitter.Result.fromH5(grp))

        return registrar

def fitFunctionLS(a, x, y):
    return skewnorm.cdf(x, *a) - y

def fitFunctionCF(x, *a):
    return skewnorm.cdf(x, *a)

def squareSum(func, indep, dep, deperr, par):
    return np.sum((func(indep, *par)-dep)**2 / deperr**2)


class Fitter:
    class Result:
        def __init__(self, bestFit, otherFits):
            self.bestFit = bestFit
            self.otherFits = otherFits

        def bestNstep(self, targetAccRate):
            return skewnorm.ppf(targetAccRate, *self.bestFit)

        def evalOn(self, x):
            return fitFunctionCF(x, *self.bestFit), \
                [fitFunctionCF(x, *params) for params in self.otherFits]

        def __eq__(self, other):
            return np.array_equal(self.bestFit, other.bestFit) \
                and np.array_equal(self.otherFits, other.otherFits)

        def save(self, h5group):
            h5group["best"] = self.bestFit
            h5group["others"] = self.otherFits

        @classmethod
        def fromH5(cls, h5group):
            return cls(h5group["best"][()],
                       h5group["others"][()])

    def __init__(self, startParams=[(0.1, 1, 10), (1, 1, 1)],
                 artificialPoints = [(0, 0.0, 1e-8), (MAX_NSTEP, 1.0, 1e-8)]):

        self._startParams = startParams
        self._lastFit = None   # parameters only
        self.artificialPoints = artificialPoints

    def _joinFitData(self, probabilityPoints, trajPointPoints):
        return np.asarray([*zip(*(probabilityPoints + trajPointPoints + self.artificialPoints))])

    def fitNstep(self, probabilityPoints, trajPointPoints):
        independent, dependent, dependenterr = self._joinFitData(probabilityPoints, trajPointPoints)
        startParams = self._startParams + (self._lastFit if self._lastFit is not None else [])

        # # w/o errors
        # result = min((least_squares(fitFunctionLS, guess, max_nfev=1000,
        #                             args=(independent, dependent),
        #                             verbose=0, loss="soft_l1", method="trf")
        #               for guess in startParams),
        #              key=lambda res: res.cost)

        # w/ errors
        results = sorted([curve_fit(fitFunctionCF, independent, dependent,
                                    p0=guess, sigma=dependenterr,
                                    absolute_sigma=True, method="trf")[0]
                          for guess in startParams],
                         key=lambda params: squareSum(fitFunctionCF, independent,
                                                      dependent, dependenterr, params))
        bestFit = results[0]
        otherFits = results[1:]

        self._lastFit = bestFit[0]

        return self.Result(bestFit, otherFits)



class LeapfrogTuner(Evolver):
    r"""! \ingroup evolvers

    """

    def __init__(self, action, initialLength, initialNstep, rng, recordFname,
                 targetAccRate=0.67, runsPerParam=(5, 100)):
        r"""!

        """

        self.registrar = Registrar(initialLength, initialNstep)
        self.action = action
        self.runsPerParam = runsPerParam
        self.targetAccRate = targetAccRate
        self.recordFname = recordFname

        self._fitter = Fitter()
        self._selector = BinarySelector(rng)
        self._finished = False

    def evolve(self, phi, pi, actVal, trajPoint):
        r"""!
        Run one step of leapfrog integration and tune parameters.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
          - Point along trajectory that was selected
        """

        # do not even evolve any more (we don't want to waste precious time)
        if self._finished:
            raise StopIteration()

        phi, pi, actVal, trajPoint = self._doEvolve(phi, pi, actVal, trajPoint)

        log = getLogger("atune")
        currentRecord = self.registrar.currentRecord()

        if len(currentRecord) > self.runsPerParam[0]:
            confIntProb = currentRecord.confIntProbabilities(TWO_SIGMA_PROB)
            confIntTP = currentRecord.confIntTrajPoints(TWO_SIGMA_PROB)

            # log.error(f"{_intervalLength(confIntProb)}, {_intervalLength(confIntTP)}")

            if _intervalLength(confIntTP) < TARGET_CONF_INT_ACC:
                log.error("Stopping because of tp")
                self._pickNextNStep()

            elif _intervalLength(confIntProb) < TARGET_CONF_INT_PROB:
                log.error("Stopping because of prob")
                self._pickNextNStep()

            elif len(currentRecord) > self.runsPerParam[1]:
                log.error("reached max runs for current params")
                # raise RuntimeError("Did not converge")
                self._pickNextNStep()

        if len(self.registrar) > 10:
            self._finalize()

        return phi, pi, actVal, trajPoint

    def currentParams(self):
        record = self.registrar.currentRecord()
        return record.length, record.nstep

    def _doEvolve(self, phi0, pi0, actVal0, _trajPoint0):
        phi1, pi1, actVal1 = leapfrog(phi0, pi0, self.action, *self.currentParams())
        energy0 = actVal0 + +np.linalg.norm(pi0)**2/2
        energy1 = actVal1 + +np.linalg.norm(pi1)**2/2
        trajPoint1 = self._selector.selectTrajPoint(energy0, energy1)

        self.registrar.currentRecord().add(min(1, exp(np.real(energy0 - energy1))),
                                           trajPoint1)

        return (phi1, pi1, actVal1, trajPoint1) if trajPoint1 == 1 \
            else (phi0, pi0, actVal0, trajPoint1)

    def _pickNextNStep(self):
        fitResult = self._fitter.fitNstep(*self.registrar.gather(
            length=self.currentParams()[0]))
        self.registrar.addFitResult(fitResult)
        floatStep = fitResult.bestNstep(self.targetAccRate)

        nextStep = max(int(floor(floatStep)), 1)
        if self.registrar.seenBefore(nstep=nextStep):
            nextStep = int(ceil(floatStep))
            # if self.registrar.seenBefore(nstep=nextStep):
                # getLogger("atune").error(f"Done with nstep = {nextStep}")
                # self._finalize()
                # return

        self.saveRecording()

        if nextStep > MAX_NSTEP:
            raise RuntimeError(f"nstep is too large: {nextStep}")

        self.registrar.newRecord(self.currentParams()[0], nextStep)
        getLogger("atune").error("New nstep: %d", nextStep)

    def _finalize(self):
        self.saveRecording()
        self._finished = True

    def saveRecording(self):
        getLogger(__name__).info("Saving current recording")
        with h5.File(self.recordFname, "a") as h5f:
            self.registrar.save(createH5Group(h5f, "leapfrogTuner"))

    @classmethod
    def loadRecording(cls, h5group):
        return Registrar.fromH5(h5group)

    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """

    @classmethod
    def fromH5(cls, h5group, _manager, action, _lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param _manager \e ignored.
        \param action Action to use.
        \param _lattice \e ignored.
        \param rng Central random number generator for the run.
        \returns A newly constructed leapfrog evolver.
        """
        return None
