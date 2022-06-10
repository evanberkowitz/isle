r"""!\file
\ingroup evolvers
Automatically tune leapfrog parameters.

Most contents of this file are for internal use, only the class LeapfrogTuner
is meant to be accessed by users.
"""

from math import sqrt, exp, floor, ceil
from logging import getLogger

import h5py as h5
import numpy as np
from scipy.stats import norm, skewnorm
from scipy.optimize import curve_fit

from .evolver import Evolver
from .selector import BinarySelector
from .leapfrog import ConstStepLeapfrog
from .transform import backwardTransform, forwardTransform
from .. import Vector, leapfrog
from ..collection import extendListInDict
from ..h5io import createH5Group, loadList


## Probability to be inside the one sigma interval of a gaussian.
ONE_SIGMA_PROB = 0.682689492
## Probability to be inside the two sigma interval of a gaussian.
TWO_SIGMA_PROB = 0.954499736


def _confIntTrajPoints(trajPoints, quantileProb):
	r"""!
	Compute a confidence interval for the trajectory points assuming a binomial distribution.
	Uses the Wald-Interval as described in
	[Vollset, S.E., 1993. 'Confidence intervals for a binomial proportion.'
	Statistics in medicine, 12(9), pp.809-824]
	This assumes that the probability to accept follows a normal distribution.

	\see _confIntProbabilities for a caveat.
	"""

	# number of accepted trajectories
	accepted = trajPoints.count(1)
	# total number of trajectories
	total = len(trajPoints)

	# quantile such that norm.cdf(quantile) == quantile_prob
	# gives the x value for the boundary of the quantileProb interval (e.g. 95% interval)
	quantile = norm.ppf(quantileProb)

	# acceptance rate
	mean = accepted / total
	# the Wald interval relative to mean
	interval = quantile/sqrt(total) * sqrt(mean*(1-mean)) + 1/(2*total)

	return mean - interval, mean + interval

def _errorTrajPoints(trajPoints, quantileProb):
	r"""!
	Compute the error for trajectory points in a certain quantile.
	\see _confIntTrajPoints for more details.
	"""
	return _intervalLength(_confIntTrajPoints(trajPoints, quantileProb)) \
		/ 2 / sqrt(len(trajPoints))

def _confIntProbabilities(probabilities, quantileProb):
	r"""!
	Compute a confidence interval for the probabilities assuming a normal distribution.
	This is not entirely correct as the probabilities are at best distributed according to
		min(1, N(mu, sigma))
	where N is a gaussian.
	But it should be close enough for tuning.
	"""

	mean = np.mean(probabilities)
	err = np.std(probabilities)

	# Sometimes, all probabilities are (almost) identical, just return a small
	# interval in order not to break later code (norm.interval would return NaN).
	if err < 1e-6:
		return mean-1e-6, mean+1e-6

	# endpoints of quantileProb confidence interval
	result = norm.interval(quantileProb, loc=mean, scale=err)
	return result[0], result[1]

def _errorProbabilities(probabilities, quantileProb):
	r"""!
	Compute the error for probabilities in a certain quantile.
	\see _confIntProbabilities for a caveat.
	"""

	# divide by two because standard deviation is only half the confidence interval
	return _intervalLength(_confIntProbabilities(probabilities, quantileProb)) \
		/ 2 / sqrt(len(probabilities))

def _intervalLength(interval):
	r"""!
	Compute the length of an interval given as interval = (lower, upper).
	"""
	return interval[1] - interval[0]


class Registrar:
	r"""!
	Keep a recording of everything that LeapfrogTuner does.

	Stores the trajectory points that are chosen after each leapfrog integration
	and the corresponding probability to accept `min(1, exp(dH))`.
	They are organized into records each of which traces runs with a specific
	set of leapfrog parameters.
	There can be multiple records with the same parameters if the tuner revisits those
	parameters after running for different ones in between.
	In addition, there is one instance of Fitter.Result per record containing the fit results
	using all records up the corresponding one (inclusive).

	This class exists mainly for internal usage by LeapfrogTuner but it can be used
	to check the tuning results from the outside.
	"""

	class Record:
		r"""!
		Hold results of runs with some fixed leapfrog parameters.
		\see Registrar for more details.
		"""

		def __init__(self, length, nstep, verification=False):
			r"""!
			Store parameters.
			\param length Trajectory length used in the run recorded here.
			\param nstep Number of integration steps used in the run recorded here.
			\param verification Is this a verification run?
			"""

			## Trajectory length (leapfrog parameter).
			self.length = length
			## Number of integration steps (leapfrog parameter).
			self.nstep = nstep
			## Recorded acceptance probabilities in the order they appeared.
			self.probabilities = []
			## Recorded trajectory points in the order they appeared.
			self.trajPoints = []
			## True if this is a verification run, False otherwise.
			self.verification = verification

		def __len__(self):
			"""!Return the number of runs that was recorded."""
			return len(self.trajPoints)

		def add(self, probability, trajPoint):
			r"""!
			Add results of a run.
			"""
			self.probabilities.append(probability)
			self.trajPoints.append(trajPoint)

		def confIntProbabilities(self, quantileProb):
			r"""!
			Compute the given confidence interval for acceptance probabilities.
			\see _confIntProbabilities for a caveat.
			"""
			return _confIntProbabilities(self.probabilities, quantileProb)

		def confIntTrajPoints(self, quantileProb):
			r"""!
			Compute the given confidence interval for trajectory points.
			\see _confIntTrajPoints for a caveat.
			"""
			return _confIntTrajPoints(self.trajPoints, quantileProb)

		def __eq__(self, other):
			"""!Check if equal to another Record."""
			return self.length == other.length \
				and self.nstep == other.nstep \
				and self.probabilities == other.probabilities \
				and self.trajPoints == other.trajPoints \
				and self.verification == other.verification

		def __str__(self):
			"""!Return a string representation."""
			return f"""Record(length={self.length}, nstep={self.nstep}, verification={self.verification}
	   probabilities={self.probabilities}
	   trajPoints={self.trajPoints})"""

		def save(self, h5group):
			"""!Save to an HDF5 group."""
			h5group["length"] = self.length
			h5group["nstep"] = self.nstep
			h5group["probabilities"] = self.probabilities
			h5group["trajPoints"] = self.trajPoints
			h5group["verification"] = self.verification

		@classmethod
		def fromH5(cls, h5group):
			"""!Construct an instance from an HDF5 group."""
			record = cls(h5group["length"][()], h5group["nstep"][()],
						 h5group["verification"][()])
			record.probabilities = list(h5group["probabilities"][()])
			record.trajPoints = list(h5group["trajPoints"][()])
			return record

	def __init__(self, initialLength, initialNstep):
		r"""!
		Set up a new recording and start with a single record with given length and nstep
		(verification is False).
		"""

		## All records in the order they were recorded, do not modify,
		## use Registrar.newRecord instead!
		self.records = []
		## All known trajectory lengths.
		self._knownLength = set()
		## All known numbers of steps.
		self._knownNstep = set()
		## Fit results, one per record.
		self.fitResults = []

		self.newRecord(initialLength, initialNstep)

	def __len__(self):
		r"""!Return the number of records."""
		return len(self.records)

	def currentRecord(self):
		r"""!Return the most recent record."""
		return self.records[-1]

	def newRecord(self, length, nstep, verification=False):
		r"""!
		Add a new record with given parameters and no recorded points.
		\returns The new record.
		"""

		record = self.Record(length, nstep, verification)
		self.records.append(record)
		self._knownLength.add(length)
		self._knownNstep.add(nstep)
		return record

	def addFitResult(self, result):
		r"""!
		Add a fit result.
		"""
		self.fitResults.append(result)

	def gather(self, *, length=None, nstep=None, maxRecord=None):
		r"""!
		Collect all acceptance probabilities and trajectory points in two lists.

		One and only one of `length` and `nstep` must be specified.

		\param length Collect all records with this trajectory length.
		\param nstep Collect all records with this number of steps.
		\param maxRecord Gather only up this record number (exclusive).

		\returns `(probabilities, trajPoints)`.
				 Each is a list (not sorted) of tuples `(param, mean, error)`, were
				 - `param` is the varying parameter, nstep if `length` was given when calling
				   this function or the other way around,
				 - `mean` is the average of all recorded trajectories for the specific parameter,
				 - `error` is the standard error on the mean.
		"""

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

		# collect in dicts to combine runs with the same parameter
		probDict = dict()
		tpDict = dict()
		for record in filter(recordFilter, self.records[:maxRecord]):
			extendListInDict(probDict, selectParam(record), record.probabilities)
			extendListInDict(tpDict, selectParam(record), record.trajPoints)

		# turn into lists of points
		probabilities = [(param, np.mean(probs), _errorProbabilities(probs, ONE_SIGMA_PROB))
						 for param, probs in probDict.items()]
		trajPoints = [(param, np.mean(tps), _errorTrajPoints(tps, ONE_SIGMA_PROB))
					  for param, tps in tpDict.items()]

		return probabilities, trajPoints

	def seenBefore(self, *, length=None, nstep=None):
		r"""!
		Check if there is a record with given length and/or nstep.
		"""

		if length is None:
			if nstep is None:
				raise ValueError("At least one of length and nstep must not be None")
			return nstep in self._knownNstep

		# else: length is not None
		if nstep is None:
			return length in self._knownLength

		# else: both not None
		return length in self._knownLength and nstep in self._knownNstep

	def knownNsteps(self):
		r"""!Return a set of all values of nstep that have been recorded."""
		return self._knownNstep.copy()

	def knownLengths(self):
		r"""!Return a set of all values of length that have been recorded."""
		return self._knownLength.copy()

	def _saveRecords(self, h5group):
		r"""!Save all records."""

		log = getLogger(__name__)

		maxStored = -1  # index of last stored record
		# check if the file is compatible with this registrar
		for idx, grp in loadList(h5group):
			if idx >= len(self.records):
				log.error("Cannot save recording, there are more records in the "
						  "file than currently recorded")
				raise RuntimeError("More records in the file that currently stored")
			storedRecord = self.Record.fromH5(grp)
			if storedRecord != self.records[idx]:
				log.error("Cannot save recording, record %d stored in the file "
						  "does dot match record in memory.", idx)
				raise RuntimeError("Record in file does not match record in memory")
			maxStored = idx

		for idx, record in filter(lambda pair: pair[0] > maxStored, enumerate(self.records)):
			if idx == len(self) and len(record) == 0:
				# the last record might be empty, do not save it
				log.info("Skipping record %d, it is empty", idx)
				break
			log.info("Saving record %d", idx)
			log.debug("Record %d = %s", idx, record)
			record.save(h5group.create_group(str(idx)))

	def _saveFitResults(self, h5group):
		r"""!Save all fit results."""

		log = getLogger(__name__)

		maxStored = -1  # index of last stored record
		# check if the file is compatible with this registrar
		for idx, grp in loadList(h5group):
			if idx >= len(self.fitResults):
				getLogger(__name__).error("Cannot save recording, there are more fit results in "
										  "the file than currently recorded")
				raise RuntimeError("More fit results in the file that currently stored")

			storedResult = Fitter.Result.fromH5(grp)
			if storedResult != self.fitResults[idx]:
				getLogger(__name__).error("Cannot save recording, fit result %d stored in the file "
										  "does dot match fit result in memory.", idx)
				raise RuntimeError("Fit result in file does not match fit result in memory")
			maxStored = idx

		for idx, fitResult in filter(lambda pair: pair[0] > maxStored, enumerate(self.fitResults)):
			log.info("Saving fit result %d", idx)
			fitResult.save(h5group.create_group(str(idx)))

	def save(self, h5group):
		r"""!
		Save all records and fit results to an HDF5 group.
		Extends existing saves if they are compatible with the data currently in memory.

		\throws RuntimeError if some data has already been saved to the file which is
							 incompatible with the current state of the registrar.
		\param h5group HDF5 group to save to.
		"""
		self._saveRecords(createH5Group(h5group, "records"))
		self._saveFitResults(createH5Group(h5group, "fitResults"))

	@classmethod
	def fromH5(cls, h5group):
		r"""!
		Construct a new Registrar from an HDF5 group.
		Retrieves the entire state of the registrar as saved by Registrar.save().
		"""

		# build a completely empty instance so we can insert the records cleanly
		registrar = cls(0, 0)
		registrar.records = []
		registrar._knownLength = set()  # pylint: disable=protected-access
		registrar._knownNstep = set()  # pylint: disable=protected-access

		for _, grp in sorted(h5group["records"].items(),
							 key=lambda pair: int(pair[0])):
			storedRecord = cls.Record.fromH5(grp)
			# go through this function to make sure all internal variables are set up properly
			record = registrar.newRecord(storedRecord.length, storedRecord.nstep,
										 storedRecord.verification)
			record.probabilities = storedRecord.probabilities
			record.trajPoints = storedRecord.trajPoints

		for _, grp in sorted(h5group["fitResults"].items(),
							 key=lambda pair: int(pair[0])):
			registrar.addFitResult(Fitter.Result.fromH5(grp))

		return registrar


def _fitFunction(x, *a):
	r"""!Function to fit to the recorded nstep."""
	return skewnorm.cdf(x, *a)

def _sumSquares(func, indep, dep, deperr, par):
	r"""!Compute a sum of squares."""
	return np.sum((func(indep, *par)-dep)**2 / deperr**2)

class Fitter:
	r"""!
	Fit a skewnormal CDF to acceptance probabilities and rates.

	Can tries several starting parameters to find the best fit.
	Arbitrary parameters can be specified in the constructor.
	In addition the results from the best previous fit are used as well.
	The best result is selected according to its sum of squares difference from the data.
	"""

	class Result:
		r"""!
		Store the parameters obtained from fits to probability and trajectory point
		versus nstep.
		"""

		def __init__(self, bestFit, otherFits):
			r"""!Store fitted parameters."""

			## List of parameters from best fit.
			self.bestFit = bestFit
			## List of lists of parameters from other fits.
			self.otherFits = otherFits

		def bestNstep(self, targetAccRate):
			r"""!Return the optimum nstep (float) for given target acceptance rate."""
			return skewnorm.ppf(targetAccRate, *self.bestFit)

		def bestLength(self, targetAccRate):
			r"""!Return the optimum length (float) for given target acceptance rate."""
			return 1./skewnorm.ppf(targetAccRate, *self.bestFit)

		def evalOn(self, x):
			r"""!Evaluate the fit function on given values with each set of fitted parameters."""
			return _fitFunction(x, *self.bestFit), \
				[_fitFunction(x, *params) for params in self.otherFits]

		def __eq__(self, other):
			"""!Check if results are equal to those in other."""
			return np.array_equal(self.bestFit, other.bestFit) \
				and np.array_equal(self.otherFits, other.otherFits)

		def save(self, h5group):
			"""!Save to an HDF5 group."""
			h5group["best"] = self.bestFit
			h5group["others"] = self.otherFits

		@classmethod
		def fromH5(cls, h5group):
			"""!Construct from an HDF5 group."""
			return cls(h5group["best"][()],
					   h5group["others"][()])

	def __init__(self, startParams=None, artificialPoints=None, maxNstep=1000):
		r"""!
		Setup a new fitter.
		\param startParams List of lists of parameters to start the fit with.
						   Each sublist must contain three parameters, the arguments
						   passed to `scipy.skewnorm.cdf`.
		\param artificialPoints List of points to insert into the fit regardless of
								measured acceptance rate or probability.
								Each element is a tuple `(nstep, value, error)`.
		\param maxNstep Maximum number of leapfrog steps per integration.
						Should be much larger than the expected optimum.
		"""

		## Initial parameters to use for fit.
		self._startParams = startParams if startParams is not None else \
			[(2, 3, 1), (1, 1, 1), (10, 2, 1)]
		## Artificial data points to insert when fitting.
		self.artificialPoints = artificialPoints if artificialPoints is not None else \
			[(0, 0.0, 1e-8), (maxNstep, 1.0, 1e-8)]
		## Parameters of previous best fit.
		self._lastFit = None

	def _joinFitData(self, probabilityPoints, trajPointPoints):
		r"""!Join data for probability, trajectory points and artifical data."""
		return np.asarray([*zip(*(probabilityPoints + trajPointPoints + self.artificialPoints))])

	def fitNstep(self, probabilityPoints, trajPointPoints):
		r"""!
		Fit a skewnormal CDF to both acceptance probability and rate.
		\returns Fitter.Result with the results from all successful fits or
				 `None` if no fit succeeded.
		"""

		# prepare inputs
		independent, dependent, dependenterr = self._joinFitData(probabilityPoints, trajPointPoints)
		startParams = self._startParams + (self._lastFit if self._lastFit is not None else [])

		fittedParams = []
		for guess in startParams:
			try:
				fittedParams.append(curve_fit(_fitFunction, independent, dependent,
											  p0=guess, sigma=dependenterr,
											  absolute_sigma=True, method="trf")[0])
			except RuntimeError as err:
				# don't save this one but continue with others
				getLogger(__name__).info("Fit failed with starting parameters %s: %s",
										 guess, err)

		if not fittedParams:
			getLogger(__name__).error("No fit converged, unable to continue tuning.")
			return None

		bestFit, *otherFits = sorted(
			fittedParams, key=lambda params: _sumSquares(_fitFunction, independent,
														 dependent, dependenterr, params))
		self._lastFit = bestFit
		return self.Result(bestFit, otherFits)
		
	def fitLength(self, probabilityPoints, trajPointPoints):
		r"""!
		Fit a skewnormal CDF to both acceptance probability and rate.
		\returns Fitter.Result with the results from all successful fits or
				 `None` if no fit succeeded.
		"""

		# prepare inputs
		invProbabilityPoints = []
		invTrajPointPoints = []
		for point in probabilityPoints:
			invProbabilityPoints.append((1./point[0],point[1],point[2]))
		for point in trajPointPoints:
			invTrajPointPoints.append((1./point[0],point[1],point[2]))
		
		independent, dependent, dependenterr = self._joinFitData(invProbabilityPoints, invTrajPointPoints)
		startParams = self._startParams + (self._lastFit if self._lastFit is not None else[])

		fittedParams = []
		for guess in startParams:
			try:
				fittedParams.append(curve_fit(_fitFunction, independent, dependent,
											  p0=guess, sigma=dependenterr,
											  absolute_sigma=True, method="trf")[0])
			except RuntimeError as err:
				# don't save this one but continue with others
				getLogger(__name__).info("Fit failed with starting parameters %s: %s",
										 guess, err)

		if not fittedParams:
			getLogger(__name__).error("No fit converged, unable to continue tuning.")
			return None

		bestFit, *otherFits = sorted(
			fittedParams, key=lambda params: _sumSquares(_fitFunction, independent,
														 dependent, dependenterr, params))
		self._lastFit = bestFit
		return self.Result(bestFit, otherFits)


class LeapfrogTuner(Evolver):  # pylint: disable=too-many-instance-attributes
	r"""! \ingroup evolvers
	Tune leapfrog parameters to achieve a targeted acceptance rate.

	This auto-tuner is based on the paper
	[Krieg et. al., 2019. 'Accelerating Hybrid Monte Carlo simulations of the Hubbard model
	on the hexagonal lattice' Comput.Phys.Commun. 236, pp.15-25].

	<B>Usage</B><BR>
	The auto-tuner can be used like any other evolver with drivers.hmc.HMC
	and tunes the leapfrog integrator while evolving the configuration.
	Ideally, you should let the tuner terminate evolution instead of specifying a
	maximum number of trajectories in the driver.
	This ensures that all results are written out correctly.

	For best results, you should start tuning with thermalized configurations.
	It has been observed that the initial nstep should best be chosen small compared to the
	expected optimum so as to stabilize the fit.
	Once tuning has completed, you can use `LeapfrogTuner.tunedEvolver()` or
	`LeapfrogTuner.tunedParameters()` to extract the tuned parameters for production.

	%LeapfrogTuner writes a detailled recording of its actions to HDF5.
	You can use the shell command `isle show -rtuning <filename>` to get an overview
	of how the tuner performed.

	\warning Do not use configurations produced by this evolver to calculate observables!
	This evolver does not produce a Markov Chain as it is not reversible!

	\attention This class does not support saving to / loading from HDF5.
	This is because the tuner decides when its recording is written to file which does
	in general not align with checkpoints written by the driver.
	Thus, it would not be possible to load the state of the tuner exactly at a checkpoint
	but only the most recently written recording.

	<B>Implementation</B><BR>
	Each time the evolve method is called, the configuration is integrated using leapfrog
	and currently selected parameters, trajectory length and number of integration steps ('nstep').
	The acceptance probability \f$\min(1, \exp(H_{\text{old}} - H_{\text{new}}))\f$ and
	trajectory point is saved for every trajectory.
	When a minimum number of runs (`runsPerParam[0]`) is reached and
	either probability or trajectory point are determined to a given precision
	(parameters `targetConfIntProb` and `targetConfIntTP`) or a maximum number of
	trajectories is reached (`runsPerParam[1]`), a new nstep is chosen.

	There are multiple stages to the tuner which affect the way nstep is selected when the above
	criterion is met.

	In the search stage, a skewnormal CDF is fitted to both the recorded probabilities and
	trajectory points simultaneously.
	If the fit is not successful, the next nstep is chosen either very small or very big compared
	to the values encountered so far in order to stabilize the fit.
	If the fit is successful however, the next nstep is chosen as `floor(PPF(target))`, where
	PPF is the inverse of the skewnormal CDF and target is the targeted acceptance rate.
	If this value has already been used before, the ceiling is taken instead.
	If that has also already been used, the tuner switches to the verification stage.

	In the verification stage, the tuner repeats calculations with both floor and ceiling
	of the previously determined optimum floating point nstep.
	If either produces an nstep which deviates from the previous by more than 1, verification
	fails and the tuner switches back to the search stage.
	Otherwise, tuning is complete.

	Once finished, the optimum nstep is calculated from all runs including verifications
	and an optimum trajectory length is estimated from a linear interpolation between
	the verification points.
	Both parameters are stored and the tuner switches to its 'finished' state.
	In this state, calling evolve() immediately raises `StopIteration` to signal
	the driver to stop.

	The different stages are implemented as a simple state maching
	by swapping out the function held by the instance variable `_pickNextNstep`.
	This function implements either the search stage (`_pickNextNstep_search()`) or
	verification (nested functions in `_enterVerification()`).
	"""

	def __init__(self, action, initialLength, initialNstep,  # pylint: disable=too-many-arguments
				 rng, recordFname, *,
				 targetAccRate=0.61, targetConfIntProb=0.125, targetConfIntTP=None,
				 maxNstep=1000, runsPerParam=(10, 100), maxRuns=12,
				 startParams=None, artificialPoints=None,
				 transform=None):
		r"""!
		Set up a leapfrog tuner.

		\param action Instance of isle.Action to use for molecular dynamics.
		\param initialLength Length of the MD trajectory.
		\param initialNstep Number of integration steps per trajectory to start running.
		\param rng Central random number generator for the run. Used for accept/reject.
		\param recordFname Name of an HDF5 file to write the recording to.

		\param targetAccRate Targeted acceptance rate.
		\param targetConfIntProb Size of the 2σ confidence interval which must be reached
								 by the acceptance probabilities in order to perform a fit
								 and change the number of MD steps.
		\param targetConfIntTP Size of the 2σ confidence interval which must be reached
							   by the trajectory points in order to perform a fit and
							   change the number of MD steps.
							   Defaults to `targetConfIntProb / 10`.
		\param maxNstep Maximum number of leapfrog steps per integration.
						Should be much larger than the expected optimum.
		\param runsPerParam Tuple (min, max)` of the minimum and maximum number of
							trajectories to compute for each set of leapfrog parameters.
		\param maxRuns Maximum number of different parameters to try.
					   If tuning did not converge at this point, LeapfrogTuner aborts.
		\param startParams List of lists of parameters to start the fits with.
						   Each sublist must contain three parameters, the arguments
						   passed to `scipy.skewnorm.cdf`.
		\param artificialPoints List of points to insert into the fit regardless of
								measured acceptance rate or probability.
								Each element is a tuple `(nstep, value, error)`.
		\param transform (Instance of isle.evolver.transform.Transform)
						 Used this to transform a configuration after MD integration
						 but before Metropolis accept/reject.
		"""

		## Record progress.
		self.registrar = Registrar(initialLength, initialNstep)
		## Action to integrate over. (*do not change!*)
		self.action = action
		## Random number generator for leapfrog evolution.
		self.rng = rng
		## Name of an HDF5 file to write the recording to.
		self.recordFname = recordFname
		## Targeted acceptance rate. (*do not change!*)
		self.targetAccRate = targetAccRate
		## Targetd size of 2σ confidence interval of acceptance probabilities.
		self.targetConfIntProb = targetConfIntProb
		## Targetd size of 2σ confidence interval of acceptance rate.
		self.targetConfIntTP = targetConfIntTP if targetConfIntTP else targetConfIntProb / 10
		## Maximum number of steps in leapfrog integration.
		self.maxNstep = maxNstep
		## Minimum and maxiumum number of runs per set of leapfrog parameters.
		self.runsPerParam = runsPerParam
		## Maxiumum number of different parameters to try.
		self.maxRuns = maxRuns
		## The transform for accept/reject.
		self.transform = transform

		## Perform fits.
		self._fitter = Fitter(startParams, artificialPoints, maxNstep)
		## Accept or reject trajectories.
		self._selector = BinarySelector(rng)
		## Pick the next nstep, is swapped out when changing stage.
		self._pickNextNstep = self._pickNextNstep_search
		## Has tuning completed?
		self._finished = False
		## Final tuned parameters, None if incomplete or unsuccessful.
		self._tunedParameters = None

	def evolve(self, stage):
		r"""!
		Run one step of leapfrog integration and tune parameters.
		\param stage EvolutionStage at the beginning of this evolution step.
		\returns EvolutionStage at the end of this evolution step.
		"""

		# do not evolve any more, signal the driver to stop
		if self._finished:
			raise StopIteration()

		stage = self._doEvolve(stage)

		log = getLogger(__name__)
		currentRecord = self.registrar.currentRecord()

		# check if the minimum number of runs has been reached
		if len(currentRecord) >= self.runsPerParam[0]:
			# get errors for current run
			errProb = _errorProbabilities(currentRecord.probabilities, TWO_SIGMA_PROB)
			errTP = _errorTrajPoints(currentRecord.trajPoints, TWO_SIGMA_PROB)

			if errTP < self.targetConfIntTP:
				log.info("Reached target confidence for trajectory point, picking next nstep")
				self._pickNextNstep()

			elif errProb < self.targetConfIntProb:
				log.info("Reached target confidence for probability, picking next nstep")
				self._pickNextNstep()

			elif len(currentRecord) > self.runsPerParam[1]:
				log.debug("Reached maximum number of runs for current nstep, picking next nstep")
				self._pickNextNstep()

		# Check here not at the beginning of the function because
		# one of the above steps may have inserted a new record.
		if not self._finished and len(self.registrar) > self.maxRuns:
			log.error("Tuning was unsuccessful within the given maximum number of runs")
			self._finalize(None)

		return stage

	def currentParams(self):
		r"""!
		Return the current (stored in most recent record) length and nstep as a dict.
		"""
		record = self.registrar.currentRecord()
		return {"length": record.length, "nstep": record.nstep}

	def _doEvolve(self, stage):
		r"""!
		Do the leapfrog integration and record probability and trajectory point.
		"""

		params = self.currentParams()

		# get start phi for MD integration
		phiMD, logdetJ = backwardTransform(self.transform, stage)
		if self.transform is not None and "logdetJ" not in stage.logWeights:
			stage.logWeights["logdetJ"] = logdetJ

		# do MD integration
		pi = Vector(self.rng.normal(0, 1, len(stage.phi))+0j)
		phiMD1, pi1, actValMD1 = leapfrog(phiMD, pi, self.action,
										  params["length"], params["nstep"])

		# transform to MC manifold
		phi1, actVal1, logdetJ1 = forwardTransform(self.transform, phiMD1, actValMD1)

		# accept/reject on MC manifold
		energy0 = stage.sumLogWeights()+np.linalg.norm(pi)**2/2
		energy1 = actVal1+logdetJ1+np.linalg.norm(pi1)**2/2
		trajPoint1 = self._selector.selectTrajPoint(energy0, energy1)

		self.registrar.currentRecord().add(min(1, exp(np.real(energy0 - energy1))),
										   trajPoint1)

		logWeights = None if self.transform is None \
			else {"logdetJ": (logdetJ, logdetJ1)[trajPoint1]}
		return stage.accept(phi1, actVal1, logWeights) if trajPoint1 == 1 \
			else stage.reject()

	def _shiftNstep(self):
		r"""!
		Double or half nstep to probe large or small acceptance rates.
		"""

		trajPoints = [trajPoint for (_, trajPoint, _)
					  in self.registrar.gather(length=self.currentParams()["length"])[1]]
		minStep = min(self.registrar.knownNsteps())
		maxStep = max(self.registrar.knownNsteps())

		# small nstep is faster => try that first
		if min(trajPoints) > 0.1:
			nextStep = max(1, minStep//2)

			# due to rounding, we might have used nextStep already
			if not self.registrar.seenBefore(nstep=nextStep):
				getLogger(__name__).debug("Shifted to small nstep: %d in run %d",
										  nextStep, len(self.registrar)-1)
				self.registrar.addFitResult(self._fitter.Result([0, 0, 0], []))
				return nextStep

		# if either check did not pass:
		if max(trajPoints) < 0.9:
			nextStep = maxStep * 2
			getLogger(__name__).debug("Shifted to large nstep: %d in run %d",
									  nextStep, len(self.registrar)-1)
			self.registrar.addFitResult(self._fitter.Result([0, 0, 0], []))
			return nextStep

		# else: try to find one in between
		nextStep = (maxStep - minStep) // 2 + minStep
		while self.registrar.seenBefore(nstep=nextStep):
			aux = (maxStep - nextStep) // 2 + nextStep
			if aux == nextStep:
				getLogger(__name__).warning("Cannot shift nstep up. Tried to shift all the way up "
											"to maximum known step and did not find any vacancies")
				# fail-safe
				return maxStep + 1
		return nextStep

	def _nstepFromFit(self):
		r"""!
		Compute the optimum nstep as a float from fitting to the current recording.
		Returns None if the fit is unsuccessful.
		"""

		log = getLogger(__name__)
		fitResult = self._fitter.fitNstep(*self.registrar.gather(
			length=self.currentParams()["length"]))

		if fitResult is not None:
			# pick nstep from fit
			log.info("Completed fit for run %d, best parameters: %s",
					 len(self.registrar)-1, fitResult.bestFit)
			self.registrar.addFitResult(fitResult)
			floatStep = fitResult.bestNstep(self.targetAccRate)
			log.info("Optimal nstep from current fit: %f", floatStep)

			return floatStep

		return None

	def _pickNextNstep_search(self):
		r"""!
		Choose a new nstep based on the entire current recording to continue
		the search for the optimum.
		Switches to the verification stage if all candidates for nstep have
		already been visited.
		"""

		log = getLogger(__name__)
		floatStep = self._nstepFromFit()
		self.saveRecording()  # save including the fit result

		if floatStep is None:
			log.info("Fit unsuccessful, shifting nstep")
			# try a different nstep at an extreme end to stabilise the fit
			nextStep = self._shiftNstep()

		else:
			# try floor or ceil
			nextStep = max(int(floor(floatStep)), 1)
			if self.registrar.seenBefore(nstep=nextStep):
				nextStep = int(ceil(floatStep))
				if self.registrar.seenBefore(nstep=nextStep):
					self._enterVerification(floatStep)
					return

		if nextStep > self.maxNstep:
			attemptedStep = nextStep
			nextStep = self.maxNstep
			while self.registrar.seenBefore(nstep=nextStep):
				if nextStep == 1:
					raise RuntimeError("Exhausted all nstep values between 1 and maximum")
				nextStep -= 1
			log.warning("Tried to use nstep=%d which is above maximum of %d. Lowered to %d",
						attemptedStep, self.maxNstep, nextStep)

		self.registrar.newRecord(self.currentParams()["length"], nextStep)
		getLogger(__name__).debug("New nstep: %d", nextStep)

	def _verificationIntStep(self, oldFloatStep):
		r"""!
		Compute an integer nstep from a fit during verification.
		Aborts verification if the new floatStep differs from the old one by more than one
		or if the fit fails.
		"""

		log = getLogger(__name__)
		floatStep = self._nstepFromFit()
		self.saveRecording()
		if floatStep is None:
			log.info("Fit unsuccessful in verification")
			self._cancelVerification(self._shiftNstep())
			return None

		if abs(floatStep-oldFloatStep) > 1:
			log.info("Nstep changed by more than 1 in verification: %d vs %d",
					 floatStep, oldFloatStep)
			self._cancelVerification(max(int(floor(floatStep)), 1))
			return None

		return floatStep

	def _enterVerification(self, floatStep):
		r"""!
		Switch to the verification stage.
		Starts a new run using floor(floatStep) and registers a new
		pickNstep which proceeds to ceil(floatStep) and potentially terminates.

		\param floatStep Floating point number for optimal nstep given current recording.
		"""

		def _pickNextNstep_verificationUpper():
			"""!Check run with upper end of interval around floatStep."""

			getLogger(__name__).debug("Checking upper end of interval around floatStep")
			nextFloatStep = self._verificationIntStep(floatStep)

			if nextFloatStep is not None:
				self._finalize(nextFloatStep)
			else:
				# something is seriously unstable if this happens
				getLogger(__name__).error("The final fit did not converge, "
										  "unable to extract nstep from tuning results. "
										  "Continuing search.")
				# verification has been canceled => do nothing more here

		def _pickNextNstep_verificationLower():
			"""!Check run with lower end of interval around floatStep."""

			getLogger(__name__).debug("Checking lower end of interval around floatStep")
			nextFloatStep = self._verificationIntStep(floatStep)

			if nextFloatStep is not None:
				# run with upper end of interval next
				self.registrar.newRecord(self.currentParams()["length"],
										 int(ceil(floatStep)),
										 True)
				self._pickNextNstep = _pickNextNstep_verificationUpper

			# else: verification has been canceled => do nothing here

		getLogger(__name__).info("Entering verification stage with nstep = %f", floatStep)
		getLogger(__name__).debug("Checking lower end of interval around floatStep")

		# run with lower end of interval next
		self.registrar.newRecord(self.currentParams()["length"],
								 max(int(floor(floatStep)), 1),
								 True)
		self._pickNextNstep = _pickNextNstep_verificationLower

	def _cancelVerification(self, nextStep):
		r"""!
		Exit verification stage and revert back to the search stage with given nstep.
		"""
		getLogger(__name__).info("Cancelling verification, reverting back to search")
		self.registrar.newRecord(self.currentParams()["length"], nextStep, False)
		self._pickNextNstep = self._pickNextNstep_search

	def _finalize(self, finalFloatStep):
		r"""!
		Wrap up after successful tuning.
		Estimate an optimum trajectory length based on given optimal nstep (float).
		Stores results in the record file.
		"""

		self._finished = True
		self.saveRecording()

		if finalFloatStep is not None:
			nstep = max(int(floor(finalFloatStep)), 1)
			# linearly interpolate between floor(floatStep) and ceil(floatStep)
			length = nstep / finalFloatStep
			self._tunedParameters = {"nstep": nstep, "length": length}

			with h5.File(self.recordFname, "a") as h5f:
				h5f["leapfrogTuner/tuned_length"] = length
				h5f["leapfrogTuner/tuned_nstep"] = nstep
			getLogger(__name__).info("Finished tuning with length = %f and nstep = %d",
									 length, nstep)

	def saveRecording(self):
		r"""!
		Save the current state of the recording. Can be incorporated into an existing save.
		"""
		getLogger(__name__).info("Saving current recording")
		with h5.File(self.recordFname, "a") as h5f:
			self.registrar.save(createH5Group(h5f, "leapfrogTuner"))

	def tunedParameters(self):
		r"""!
		Return the tuned length and nstep is available.
		\throws RuntimeError if tuning is not complete/successful.
		\returns `dict` with keys `'length'` and `'nstep'`.
		"""

		if not self._finished:
			raise RuntimeError("LeapfrogTuner has not finished, parameters have not been tuned")
		if not self._tunedParameters:
			raise RuntimeError("LeapfrogTuner has finished but parameters could not be tuned")

		return self._tunedParameters.copy()

	def tunedEvolver(self, rng=None):
		r"""!
		Construct a new leapfrog evolver with tuned parameters.
		\param rng Use this RNG for the evolver or use the one passed to the constructor of the
			   tuner if `rng is None`.
		\throws RuntimeError if tuning is not complete/successful.
		\returns A new instance of evolver.leapfrog.ConstStepLeapfrog with
				 the tuned length and nstep.
		"""
		params = self.tunedParameters()
		return ConstStepLeapfrog(self.action,
								 params["length"],
								 params["nstep"],
								 self._selector.rng if rng is None else rng,
								 transform=self.transform)

	@classmethod
	def loadTunedParameters(cls, h5group):
		r"""!
		Load tuned parameters from HDF5.
		\param h5group Base group that contains the tuner group, i.e.
					   `h5group['leapfrogTuner']` must exist.
		\throws RuntimeError if tuning was not complete/successful when the tuner was last saved.
		\returns `dict` with keys `'length'` and `'nstep'`.
		"""
		h5group = h5group["leapfrogTuner"]

		if "tuned_length" not in h5group or "tuned_nstep" not in h5group:
			raise RuntimeError("LeapfrogTuner has not finished, parameters have not been tuned")

		return {"length": h5group["tuned_length"][()],
				"nstep": h5group["tuned_nstep"][()]}

	@classmethod
	def loadTunedEvolver(cls, h5group, action, rng):
		r"""!
		Construct a new leapfrog evolver with tuned parameters loaded from HDF5.
		\param h5group Base group that contains the tuner group, i.e.
					   `h5group['leapfrogTuner']` must exist.
		\param action Instance of isle.Action to use for molecular dynamics.
		\param rng Central random number generator for the run. Used for accept/reject.
		\throws RuntimeError if tuning is not complete/successful.
		\returns A new instance of evolver.leapfrog.ConstStepLeapfrog with
				 the tuned length and nstep.
		"""
		params = cls.loadTunedParameters(h5group)
		return ConstStepLeapfrog(action, params["length"],
								 params["nstep"], rng)

	@classmethod
	def loadRecording(cls, h5group):
		r"""!
		Load a recording from HDF5.
		\returns A new instance of Registrar.
		"""
		return Registrar.fromH5(h5group)

	def save(self, h5group, manager):
		r"""!
		Save the evolver to HDF5.
		\param h5group HDF5 group to save to.
		\param manager EvolverManager whose purview to save the evolver in.
		"""
		raise NotImplementedError("Saving to HDF5 is not supported.")

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
		raise NotImplementedError("Loading from HDF5 is not supported.")

	def report(self):
		r"""!
		Return a string summarizing the evolution since the evolver
		was constructed including by fromH5.
		"""
		return f"""<Autotuner> (0x{id(self):x})
  record file = {self.recordFname}"""

class LeapfrogTunerLength(Evolver):  # pylint: disable=too-many-instance-attributes
	r"""! \ingroup evolvers
	Tune leapfrog parameters to achieve a targeted acceptance rate.

	This auto-tuner is based on the paper
	[Krieg et. al., 2019. 'Accelerating Hybrid Monte Carlo simulations of the Hubbard model
	on the hexagonal lattice' Comput.Phys.Commun. 236, pp.15-25].

	<B>Usage</B><BR>
	The auto-tuner can be used like any other evolver with drivers.hmc.HMC
	and tunes the leapfrog integrator while evolving the configuration.
	Ideally, you should let the tuner terminate evolution instead of specifying a
	maximum number of trajectories in the driver.
	This ensures that all results are written out correctly.

	For best results, you should start tuning with thermalized configurations.
	It has been observed that the initial nstep should best be chosen small compared to the
	expected optimum so as to stabilize the fit.
	Once tuning has completed, you can use `LeapfrogTuner.tunedEvolver()` or
	`LeapfrogTuner.tunedParameters()` to extract the tuned parameters for production.

	%LeapfrogTuner writes a detailled recording of its actions to HDF5.
	You can use the shell command `isle show -rtuning <filename>` to get an overview
	of how the tuner performed.

	\warning Do not use configurations produced by this evolver to calculate observables!
	This evolver does not produce a Markov Chain as it is not reversible!

	\attention This class does not support saving to / loading from HDF5.
	This is because the tuner decides when its recording is written to file which does
	in general not align with checkpoints written by the driver.
	Thus, it would not be possible to load the state of the tuner exactly at a checkpoint
	but only the most recently written recording.

	<B>Implementation</B><BR>
	Each time the evolve method is called, the configuration is integrated using leapfrog
	and currently selected parameters, trajectory length and number of integration steps ('nstep').
	The acceptance probability \f$\min(1, \exp(H_{\text{old}} - H_{\text{new}}))\f$ and
	trajectory point is saved for every trajectory.
	When a minimum number of runs (`runsPerParam[0]`) is reached and
	either probability or trajectory point are determined to a given precision
	(parameters `targetConfIntProb` and `targetConfIntTP`) or a maximum number of
	trajectories is reached (`runsPerParam[1]`), a new nstep is chosen.

	There are multiple stages to the tuner which affect the way nstep is selected when the above
	criterion is met.

	In the search stage, a skewnormal CDF is fitted to both the recorded probabilities and
	trajectory points simultaneously.
	If the fit is not successful, the next nstep is chosen either very small or very big compared
	to the values encountered so far in order to stabilize the fit.
	If the fit is successful however, the next nstep is chosen as `floor(PPF(target))`, where
	PPF is the inverse of the skewnormal CDF and target is the targeted acceptance rate.
	If this value has already been used before, the ceiling is taken instead.
	If that has also already been used, the tuner switches to the verification stage.

	In the verification stage, the tuner repeats calculations with both floor and ceiling
	of the previously determined optimum floating point nstep.
	If either produces an nstep which deviates from the previous by more than 1, verification
	fails and the tuner switches back to the search stage.
	Otherwise, tuning is complete.

	Once finished, the optimum nstep is calculated from all runs including verifications
	and an optimum trajectory length is estimated from a linear interpolation between
	the verification points.
	Both parameters are stored and the tuner switches to its 'finished' state.
	In this state, calling evolve() immediately raises `StopIteration` to signal
	the driver to stop.

	The different stages are implemented as a simple state maching
	by swapping out the function held by the instance variable `_pickNextNstep`.
	This function implements either the search stage (`_pickNextNstep_search()`) or
	verification (nested functions in `_enterVerification()`).
	"""

	def __init__(self, action, initialLength, Nstep,  # pylint: disable=too-many-arguments
				 rng, recordFname, *,
				 targetAccRate=0.7, targetConfIntProb=0.01, targetConfIntTP=None,
				 maxLength=1000, runsPerParam=(2000, 2000), maxRuns=50,
				 startParams=None, artificialPoints=None,
				 transform=None):
		r"""!
		Set up a leapfrog tuner.

		\param action Instance of isle.Action to use for molecular dynamics.
		\param initialLength Length of the MD trajectory.
		\param initialNstep Number of integration steps per trajectory to start running.
		\param rng Central random number generator for the run. Used for accept/reject.
		\param recordFname Name of an HDF5 file to write the recording to.

		\param targetAccRate Targeted acceptance rate.
		\param targetConfIntProb Size of the 2σ confidence interval which must be reached
								 by the acceptance probabilities in order to perform a fit
								 and change the number of MD steps.
		\param targetConfIntTP Size of the 2σ confidence interval which must be reached
							   by the trajectory points in order to perform a fit and
							   change the number of MD steps.
							   Defaults to `targetConfIntProb / 10`.
		\param maxLength Maximum inverse(!) length of leapfrog steps.
						Should be much larger than the expected optimum.
		\param runsPerParam Tuple (min, max)` of the minimum and maximum number of
							trajectories to compute for each set of leapfrog parameters.
		\param maxRuns Maximum number of different parameters to try.
					   If tuning did not converge at this point, LeapfrogTuner aborts.
		\param startParams List of lists of parameters to start the fits with.
						   Each sublist must contain three parameters, the arguments
						   passed to `scipy.skewnorm.cdf`.
		\param artificialPoints List of points to insert into the fit regardless of
								measured acceptance rate or probability.
								Each element is a tuple `(nstep, value, error)`.
		\param transform (Instance of isle.evolver.transform.Transform)
						 Used this to transform a configuration after MD integration
						 but before Metropolis accept/reject.
		"""

		## Record progress.
		self.registrar = Registrar(initialLength, Nstep)
		## Action to integrate over. (*do not change!*)
		self.action = action
		## Random number generator for leapfrog evolution.
		self.rng = rng
		## Name of an HDF5 file to write the recording to.
		self.recordFname = recordFname
		## Targeted acceptance rate. (*do not change!*)
		self.targetAccRate = targetAccRate
		## Targetd size of 2σ confidence interval of acceptance probabilities.
		self.targetConfIntProb = targetConfIntProb
		## Targetd size of 2σ confidence interval of acceptance rate.
		self.targetConfIntTP = targetConfIntTP if targetConfIntTP else targetConfIntProb / 10
		## Maximum number of steps in leapfrog integration.
		self.maxLength = maxLength
		## Minimum and maxiumum number of runs per set of leapfrog parameters.
		self.runsPerParam = runsPerParam
		## Maxiumum number of different parameters to try.
		self.maxRuns = maxRuns
		## The transform for accept/reject.
		self.transform = transform

		## Perform fits.
		self._fitter = Fitter(startParams, artificialPoints, 1000)
		## Accept or reject trajectories.
		self._selector = BinarySelector(rng)
		## Pick the next nstep, is swapped out when changing stage.
		self._pickNextLength = self._pickNextLength_search
		## Has tuning completed?
		self._finished = False
		## Final tuned parameters, None if incomplete or unsuccessful.
		self._tunedParameters = None

	def evolve(self, stage):
		r"""!
		Run one step of leapfrog integration and tune parameters.
		\param stage EvolutionStage at the beginning of this evolution step.
		\returns EvolutionStage at the end of this evolution step.
		"""

		# do not evolve any more, signal the driver to stop
		if self._finished:
			raise StopIteration()

		stage = self._doEvolve(stage)

		log = getLogger(__name__)
		currentRecord = self.registrar.currentRecord()

		# check if the minimum number of runs has been reached
		if len(currentRecord) >= self.runsPerParam[0]:
			# get errors for current run
			errProb = _errorProbabilities(currentRecord.probabilities, TWO_SIGMA_PROB)
			errTP = _errorTrajPoints(currentRecord.trajPoints, TWO_SIGMA_PROB)

			if errTP < self.targetConfIntTP:
				log.info("Reached target confidence for trajectory point, picking next Length")
				self._pickNextLength()

			elif errProb < self.targetConfIntProb:
				log.info("Reached target confidence for probability, picking next Length")
				self._pickNextLength()

			elif len(currentRecord) > self.runsPerParam[1]:
				log.debug("Reached maximum number of runs for current nstep, picking next Length")
				self._pickNextLength()

		# Check here not at the beginning of the function because
		# one of the above steps may have inserted a new record.
		if not self._finished and len(self.registrar) > self.maxRuns:
			log.error("Tuning was unsuccessful within the given maximum number of runs")
			self._finalize(None)

		return stage

	def currentParams(self):
		r"""!
		Return the current (stored in most recent record) length and nstep as a dict.
		"""
		record = self.registrar.currentRecord()
		return {"length": record.length, "nstep": record.nstep}

	def _doEvolve(self, stage):
		r"""!
		Do the leapfrog integration and record probability and trajectory point.
		"""

		params = self.currentParams()

		# get start phi for MD integration
		phiMD, logdetJ = backwardTransform(self.transform, stage)
		if self.transform is not None and "logdetJ" not in stage.logWeights:
			stage.logWeights["logdetJ"] = logdetJ

		# do MD integration
		pi = Vector(self.rng.normal(0, 1, len(stage.phi))+0j)
		phiMD1, pi1, actValMD1 = leapfrog(phiMD, pi, self.action,
										  params["length"], params["nstep"])

		# transform to MC manifold
		phi1, actVal1, logdetJ1 = forwardTransform(self.transform, phiMD1, actValMD1)

		# accept/reject on MC manifold
		energy0 = stage.sumLogWeights()+np.linalg.norm(pi)**2/2
		energy1 = actVal1+logdetJ1+np.linalg.norm(pi1)**2/2
		trajPoint1 = self._selector.selectTrajPoint(energy0, energy1)

		self.registrar.currentRecord().add(min(1, exp(np.real(energy0 - energy1))),
										   trajPoint1)

		logWeights = None if self.transform is None \
			else {"logdetJ": (logdetJ, logdetJ1)[trajPoint1]}
		return stage.accept(phi1, actVal1, logWeights) if trajPoint1 == 1 \
			else stage.reject()

	def _shiftLength(self):
		r"""!
		Double or half length to probe large or small acceptance rates.
		"""

		trajPoints = [trajPoint for (_, trajPoint, _)
					  in self.registrar.gather(nstep=self.currentParams()["nstep"])[1]]         #? 0/1
		minLength = min(self.registrar.knownLengths())
		maxLength = max(self.registrar.knownLengths())

		# small nstep is faster => try that first
		if min(trajPoints) > 0.1:
			nextLength = maxLength * 2

			# due to rounding, we might have used nextStep already
			if not self.registrar.seenBefore(length=nextLength):
				getLogger(__name__).debug("Shifted to large length: %d in run %d",
										  nextLength, len(self.registrar)-1)
				self.registrar.addFitResult(self._fitter.Result([0, 0, 0], []))
				return nextLength

		# if either check did not pass:
		if max(trajPoints) < 0.9:
			nextLength = minLength / 2
			getLogger(__name__).debug("Shifted to smaller length: %f in run %d",
									  nextLength, len(self.registrar)-1)
			self.registrar.addFitResult(self._fitter.Result([0, 0, 0], []))
			return nextLength

		# else: try to find one in between
		nextLength = (maxLength - minLength) / 2 + minLength
		while self.registrar.seenBefore(length=nextLength):
			aux = (nextLength - minLength) / 2 + minLength
			if aux == nextLength:
				getLogger(__name__).warning("Cannot shift nstep up. Tried to shift all the way up "
											"to maximum known step and did not find any vacancies")
				# fail-safe
				return maxLength + 1
		return nextLength

	def _lengthFromFit(self):
		r"""!
		Compute the optimum length as a float from fitting to the current recording.
		Returns None if the fit is unsuccessful.
		"""

		log = getLogger(__name__)
		fitResult = self._fitter.fitLength(*self.registrar.gather(
			nstep=self.currentParams()["nstep"]))

		if fitResult is not None:
			# pick length from fit
			log.info("Completed fit for run %d, best parameters: %s",
					 len(self.registrar)-1, fitResult.bestFit)
			self.registrar.addFitResult(fitResult)
			length = fitResult.bestLength(self.targetAccRate)
			log.info("Optimal length from current fit: %f", length)

			return length

		return None

	def _pickNextLength_search(self):
		r"""!
		Choose a new length based on the entire current recording to continue
		the search for the optimum.
		Switches to the verification stage if all candidates for length have
		already been visited.
		"""

		log = getLogger(__name__)#
		#length = self.currentParams()["length"]
		length = self._lengthFromFit()
		self.saveRecording()  # save including the fit result

		if length is None:
			log.info("Fit unsuccessful, shifting length")
			# try a different length at an extreme end to stabilise the fit
			nextLength = self._shiftLength()

		else:
			nextLength = length
			acceptanceRate = np.mean(self.registrar.currentRecord().trajPoints)
			if abs(self.currentParams()["length"]/nextLength - 1) < 0.1 and abs(self.targetAccRate - acceptanceRate) < 0.025:
				self._enterVerification(nextLength)
				return

		if nextLength > self.maxLength:
			attemptedLength = nextLength
			nextLength = self.maxLength
			while self.registrar.seenBefore(length=nextLength):
				nextLength -= 1
			log.warning("Tried to use length=%f which is above maximum of %f. Lowered to %f",
						attemptedLength, self.maxLength, nextLength)

		self.registrar.newRecord(nextLength, self.currentParams()["nstep"])
		getLogger(__name__).debug("New length: %f", nextLength)

	def _verificationLength(self, oldLength):
		r"""!
		Compute length from a fit during verification.
		Aborts verification if the new length differs from the old one by more than 0.01
		or if the fit fails.
		"""

		log = getLogger(__name__)
		length = self._lengthFromFit()
		acceptanceRate = np.mean(self.registrar.currentRecord().trajPoints)
		self.saveRecording()
		if length is None:
			log.info("Fit unsuccessful in verification")
			self._cancelVerification(self._shiftLength())
			return None

		if abs(length/oldLength-1) > 0.05 or abs(acceptanceRate - self.targetAccRate) > 0.025:
			log.info("length changed by more than 5%% in verification: %f vs %f\n or target acceptance rate missed by more that 0.025: %f vs %f",
					 length, oldLength, self.targetAccRate, acceptanceRate)
			self._cancelVerification(length)
			return None
		log.info("acceptance rate = %f",acceptanceRate)
		return length

	def _enterVerification(self, length):
		r"""!
		Switch to the verification stage.
		Starts a new run using length and registers a new
		pickLength which proceeds to length*0.9 and potentially terminates.

		\param length Floating point number for optimal length given current recording.
		"""
		getLogger(__name__).info("Entering verification stage with length = %f", length)
		self.runsPerParam = tuple([4*x for x in self.runsPerParam])
		def _pickNextLength_verification():
			"""!Check run with lower end of interval around floatStep."""

			getLogger(__name__).debug("Checking upper end of interval around floatStep")
			nextLength = self._verificationLength(length)

			if nextLength is not None:
				self._finalize(nextLength)
			else:
				# something is seriously unstable if this happens
				getLogger(__name__).error("The final fit did not converge, "
										  "unable to extract nstep from tuning results. "
										  "Continuing search.")
				# verification has been canceled => do nothing more here


		# run with length next
		self.registrar.newRecord(length, self.currentParams()["nstep"],
								 True)
		self._pickNextLength = _pickNextLength_verification

	def _cancelVerification(self, nextLength):
		r"""!
		Exit verification stage and revert back to the search stage with given length.
		"""
		getLogger(__name__).info("Cancelling verification, reverting back to search")
		self.runsPerParam = tuple([x/4 for x in self.runsPerParam])
		self.registrar.newRecord(nextLength, self.currentParams()["nstep"], False)
		self._pickNextLength = self._pickNextLength_search

	def _finalize(self, finalLength):
		r"""!
		Wrap up after successful tuning.
		Stores results in the record file.
		"""

		self._finished = True
		self.saveRecording()

		if finalLength is not None:
			nstep = self.currentParams()["nstep"]
			length = finalLength
			self._tunedParameters = {"nstep": nstep, "length": length}

			with h5.File(self.recordFname, "a") as h5f:
				h5f["leapfrogTuner/tuned_length"] = length
				h5f["leapfrogTuner/tuned_nstep"] = nstep
			getLogger(__name__).info("Finished tuning with length = %f and nstep = %d",
									 length, nstep)

	def saveRecording(self):
		r"""!
		Save the current state of the recording. Can be incorporated into an existing save.
		"""
		getLogger(__name__).info("Saving current recording")
		with h5.File(self.recordFname, "a") as h5f:
			self.registrar.save(createH5Group(h5f, "leapfrogTuner"))

	def tunedParameters(self):
		r"""!
		Return the tuned length and nstep is available.
		\throws RuntimeError if tuning is not complete/successful.
		\returns `dict` with keys `'length'` and `'nstep'`.
		"""

		if not self._finished:
			raise RuntimeError("LeapfrogTuner has not finished, parameters have not been tuned")
		if not self._tunedParameters:
			raise RuntimeError("LeapfrogTuner has finished but parameters could not be tuned")

		return self._tunedParameters.copy()

	def tunedEvolver(self, rng=None):
		r"""!
		Construct a new leapfrog evolver with tuned parameters.
		\param rng Use this RNG for the evolver or use the one passed to the constructor of the
			   tuner if `rng is None`.
		\throws RuntimeError if tuning is not complete/successful.
		\returns A new instance of evolver.leapfrog.ConstStepLeapfrog with
				 the tuned length and nstep.
		"""
		params = self.tunedParameters()
		return ConstStepLeapfrog(self.action,
								 params["length"],
								 params["nstep"],
								 self._selector.rng if rng is None else rng,
								 transform=self.transform)

	@classmethod
	def loadTunedParameters(cls, h5group):
		r"""!
		Load tuned parameters from HDF5.
		\param h5group Base group that contains the tuner group, i.e.
					   `h5group['leapfrogTuner']` must exist.
		\throws RuntimeError if tuning was not complete/successful when the tuner was last saved.
		\returns `dict` with keys `'length'` and `'nstep'`.
		"""
		h5group = h5group["leapfrogTuner"]

		if "tuned_length" not in h5group or "tuned_nstep" not in h5group:
			raise RuntimeError("LeapfrogTuner has not finished, parameters have not been tuned")

		return {"length": h5group["tuned_length"][()],
				"nstep": h5group["tuned_nstep"][()]}

	@classmethod
	def loadTunedEvolver(cls, h5group, action, rng, trafo=None):
		r"""!
		Construct a new leapfrog evolver with tuned parameters loaded from HDF5.
		\param h5group Base group that contains the tuner group, i.e.
					   `h5group['leapfrogTuner']` must exist.
		\param action Instance of isle.Action to use for molecular dynamics.
		\param rng Central random number generator for the run. Used for accept/reject.
		\throws RuntimeError if tuning is not complete/successful.
		\returns A new instance of evolver.leapfrog.ConstStepLeapfrog with
				 the tuned length and nstep.
		"""
		params = cls.loadTunedParameters(h5group)
		return ConstStepLeapfrog(action, params["length"],
								 params["nstep"], rng, transform=trafo)

	@classmethod
	def loadRecording(cls, h5group):
		r"""!
		Load a recording from HDF5.
		\returns A new instance of Registrar.
		"""
		return Registrar.fromH5(h5group)

	def save(self, h5group, manager):
		r"""!
		Save the evolver to HDF5.
		\param h5group HDF5 group to save to.
		\param manager EvolverManager whose purview to save the evolver in.
		"""
		raise NotImplementedError("Saving to HDF5 is not supported.")

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
		raise NotImplementedError("Loading from HDF5 is not supported.")

	def report(self):
		r"""!
		Return a string summarizing the evolution since the evolver
		was constructed including by fromH5.
		"""
		return f"""<Autotuner> (0x{id(self):x})
  record file = {self.recordFname}"""

class LeapfrogTunerLength2(Evolver):  # pylint: disable=too-many-instance-attributes

	def __init__(self, action, initialLength, Nstep,  # pylint: disable=too-many-arguments
				 rng, recordFname, *,
				 targetAccRate=0.7, targetConfIntProb=0.01, targetConfIntTP=None,
				 maxLength=100, runsPerParam=(2000, 2000), maxRuns=50,
				 artificialPoints=None,
				 transform=None):
		
		self.registrar = Registrar(initialLength, Nstep)
		self.action = action
		self.rng = rng
		self.recordFname = recordFname
		self.targetAccRate = targetAccRate
		self.targetConfIntProb = targetConfIntProb
		self.targetConfIntTP = targetConfIntTP if targetConfIntTP else targetConfIntProb / 10
		self.maxLength = maxLength
		self.runsPerParam = runsPerParam
		self.maxRuns = maxRuns
		self.transform = transform

		self._bounds = [0.0, 2*initialLength]
		self._selector = BinarySelector(rng)
		self._pickNextLength = self._pickNextLength_search
		self._finished = False
		self._tunedParameters = None

	def evolve(self, stage):
		if self._finished:
			raise StopIteration()

		stage = self._doEvolve(stage)

		log = getLogger(__name__)
		currentRecord = self.registrar.currentRecord()

		if len(currentRecord) >= self.runsPerParam[0]:
			errProb = _errorProbabilities(currentRecord.probabilities, TWO_SIGMA_PROB)
			errTP = _errorTrajPoints(currentRecord.trajPoints, TWO_SIGMA_PROB)

			if errTP < self.targetConfIntTP:
				log.info("Reached target confidence for trajectory point, picking next Length")
				self._pickNextLength()

			elif errProb < self.targetConfIntProb:
				log.info("Reached target confidence for probability, picking next Length")
				self._pickNextLength()

			elif len(currentRecord) > self.runsPerParam[1]:
				log.debug("Reached maximum number of runs for current nstep, picking next Length")
				self._pickNextLength()


		if not self._finished and len(self.registrar) > self.maxRuns:
			log.error("Tuning was unsuccessful within the given maximum number of runs")
			self._finalize(None)

		return stage

	def currentParams(self):
		record = self.registrar.currentRecord()
		return {"length": record.length, "nstep": record.nstep}

	def _doEvolve(self, stage):
		params = self.currentParams()

		phiMD, logdetJ = backwardTransform(self.transform, stage)
		if self.transform is not None and "logdetJ" not in stage.logWeights:
			stage.logWeights["logdetJ"] = logdetJ

		pi = Vector(self.rng.normal(0, 1, len(stage.phi))+0j)
		phiMD1, pi1, actValMD1 = leapfrog(phiMD, pi, self.action,
										  params["length"], params["nstep"])

		phi1, actVal1, logdetJ1 = forwardTransform(self.transform, phiMD1, actValMD1)

		energy0 = stage.sumLogWeights()+np.linalg.norm(pi)**2/2
		energy1 = actVal1+logdetJ1+np.linalg.norm(pi1)**2/2
		trajPoint1 = self._selector.selectTrajPoint(energy0, energy1)

		self.registrar.currentRecord().add(min(1, exp(np.real(energy0 - energy1))),
										   trajPoint1)

		logWeights = None if self.transform is None \
			else {"logdetJ": (logdetJ, logdetJ1)[trajPoint1]}
		return stage.accept(phi1, actVal1, logWeights) if trajPoint1 == 1 \
			else stage.reject()

	def _shiftLength(self):
		trajPoints = [trajPoint for (_, trajPoint, _)
					  in self.registrar.gather(nstep=self.currentParams()["nstep"])[1]]         #? 0/1
		minLength = min(self.registrar.knownLengths())
		maxLength = max(self.registrar.knownLengths())

		if min(trajPoints) > 0.1:
			nextLength = maxLength * 2

			if not self.registrar.seenBefore(length=nextLength):
				getLogger(__name__).debug("Shifted to large length: %d in run %d",
										  nextLength, len(self.registrar)-1)
				self.registrar.addFitResult(self._fitter.Result([0, 0, 0], []))
				return nextLength

		if max(trajPoints) < 0.9:
			nextLength = minLength / 2
			getLogger(__name__).debug("Shifted to smaller length: %f in run %d",
									  nextLength, len(self.registrar)-1)
			self.registrar.addFitResult(self._fitter.Result([0, 0, 0], []))
			return nextLength

		nextLength = (maxLength - minLength) / 2 + minLength
		while self.registrar.seenBefore(length=nextLength):
			aux = (nextLength - minLength) / 2 + minLength
			if aux == nextLength:
				getLogger(__name__).warning("Cannot shift nstep up. Tried to shift all the way up "
											"to maximum known step and did not find any vacancies")
				# fail-safe
				return maxLength + 1
		return nextLength

	def currentBounds(self):
		return self._bounds

	def _lengthFromBisection(self):
		r"""!
		Compute the optimum length as a float from fitting to the current recording.
		Returns None if the fit is unsuccessful.
		"""

		log = getLogger(__name__)
		upper_length, lower_length = self.currentBounds()
		new_length = (upper_length + lower_length)/2

		if new_length is not None:
			# pick length from fit
			log.info("Chose new length for run %d, new length: %s",
					 len(self.registrar)-1, new_length)
			# self.registrar.addFitResult(new_length)			#TODO
			length = new_length
			log.info("Optimal length from current iteration: %f", length)

			return length

		return None

	def _pickNextLength_search(self):
		log = getLogger(__name__)#
		self.saveRecording()  # save including the fit result

		length = self.currentParams()["length"]
		acceptanceRate = np.mean(self.registrar.currentRecord().probabilities)
		log.info(f"{length=}   {acceptanceRate=}")
		if acceptanceRate < self.targetAccRate:
			self._bounds[1] = length
		else:
			self._bounds[0] = length
		log.info("acceptance rate was %f. New bounds for bisection: [ %f , %f ]", acceptanceRate, self._bounds[0],self._bounds[1])

		nextLength = self._lengthFromBisection()

		if abs(self.currentParams()["length"]/nextLength - 1) < 0.2 and abs(self.targetAccRate - acceptanceRate) < 0.05:
			self._enterVerification(nextLength)
			return

		if nextLength > self.maxLength:
			attemptedLength = nextLength
			nextLength = self.maxLength
			while self.registrar.seenBefore(length=nextLength):
				nextLength -= 1
			log.warning("Tried to use length=%f which is above maximum of %f. Lowered to %f",
						attemptedLength, self.maxLength, nextLength)

		self.registrar.newRecord(nextLength, self.currentParams()["nstep"])
		getLogger(__name__).debug("New length: %f", nextLength)

	def _verificationLength(self, oldLength):
		log = getLogger(__name__)
		length = self._lengthFromBisection()
		acceptanceRate = np.mean(self.registrar.currentRecord().probabilities)
		self.saveRecording()
		if length is None:
			log.info("Fit unsuccessful in verification")
			self._cancelVerification(self._shiftLength())
			return None

		if abs(length/oldLength-1) > 0.05 or abs(acceptanceRate - self.targetAccRate) > 0.025:
			log.info("length changed by more than 5%% in verification: %f vs %f\n or target acceptance rate missed by more that 0.025: %f vs %f",
					 length, oldLength, self.targetAccRate, acceptanceRate)
			if acceptanceRate < self.targetAccRate:
				log.info("acceptance rate too small. Reducing lower bound: %f -> %f", self._bounds[0],self._bounds[0]*0.9)
				self._bounds[0]*=0.9
			else:
				log.info("acceptance rate too high. Increasing upper bound: %f -> %f", self._bounds[1],self._bounds[1]*1.1)
				self._bounds[1]*=1.1
			self._cancelVerification(length)
			return None
		log.info("acceptance rate = %f",acceptanceRate)
		return length

	def _enterVerification(self, length):
		getLogger(__name__).info("Entering verification stage with length = %f", length)
		self.runsPerParam = tuple([4*x for x in self.runsPerParam])
		def _pickNextLength_verification():
			getLogger(__name__).debug("Checking upper end of interval around floatStep")
			nextLength = self._verificationLength(length)

			if nextLength is not None:
				self._finalize(nextLength)
			else:
				# something is seriously unstable if this happens
				getLogger(__name__).error("The final fit did not converge, "
										  "unable to extract nstep from tuning results. "
										  "Continuing search.")

		self.registrar.newRecord(length, self.currentParams()["nstep"],
								 True)
		self._pickNextLength = _pickNextLength_verification

	def _cancelVerification(self, nextLength):
		getLogger(__name__).info("Cancelling verification, reverting back to search")
		self.runsPerParam = tuple([x/4 for x in self.runsPerParam])
		self.registrar.newRecord(nextLength, self.currentParams()["nstep"], False)
		self._pickNextLength = self._pickNextLength_search

	def _finalize(self, finalLength):
		self._finished = True
		self.saveRecording()

		if finalLength is not None:
			nstep = self.currentParams()["nstep"]
			length = finalLength
			self._tunedParameters = {"nstep": nstep, "length": length}

			with h5.File(self.recordFname, "a") as h5f:
				h5f["leapfrogTuner/tuned_length"] = length
				h5f["leapfrogTuner/tuned_nstep"] = nstep
			getLogger(__name__).info("Finished tuning with length = %f and nstep = %d",
									 length, nstep)

	def saveRecording(self):
		getLogger(__name__).info("Saving current recording")
		with h5.File(self.recordFname, "a") as h5f:
			self.registrar.save(createH5Group(h5f, "leapfrogTuner"))

	def tunedParameters(self):
		if not self._finished:
			raise RuntimeError("LeapfrogTuner has not finished, parameters have not been tuned")
		if not self._tunedParameters:
			raise RuntimeError("LeapfrogTuner has finished but parameters could not be tuned")

		return self._tunedParameters.copy()

	def tunedEvolver(self, rng=None):
		params = self.tunedParameters()
		return ConstStepLeapfrog(self.action,
								 params["length"],
								 params["nstep"],
								 self._selector.rng if rng is None else rng,
								 transform=self.transform)

	@classmethod
	def loadTunedParameters(cls, h5group):
		h5group = h5group["leapfrogTuner"]

		if "tuned_length" not in h5group or "tuned_nstep" not in h5group:
			raise RuntimeError("LeapfrogTuner has not finished, parameters have not been tuned")

		return {"length": h5group["tuned_length"][()],
				"nstep": h5group["tuned_nstep"][()]}

	@classmethod
	def loadTunedEvolver(cls, h5group, action, rng, trafo=None):
		params = cls.loadTunedParameters(h5group)
		return ConstStepLeapfrog(action, params["length"],
								 params["nstep"], rng, transform=trafo)

	@classmethod
	def loadRecording(cls, h5group):
		return Registrar.fromH5(h5group)

	def save(self, h5group, manager):
		raise NotImplementedError("Saving to HDF5 is not supported.")

	@classmethod
	def fromH5(cls, h5group, _manager, action, _lattice, rng):
		raise NotImplementedError("Loading from HDF5 is not supported.")

	def report(self):
		return f"""<Autotuner> (0x{id(self):x})
	record file = {self.recordFname}"""