r"""!\file
\ingroup evolvers
Store a state of MCMC evolution.
"""


from isle import Vector


class EvolutionStage:
    """!
    Hold the status of MCMC at some point in the evolution.

    Stores the configuration and corresponding weight.
    `exp(-real(stage.sumLogWeights()))` is the weight used for accept/reject
    and the correspondint imagianry part is used for re-weighting.
    Not the minus sign!
    """
    # TODO is the minus correct?

    __slots__ = "phi", "trajPoint", "logWeights", "extra"

    def __init__(self, phi, actVal, trajPoint=1, logWeights=None, extra=None):
        """!
        Store parameters.
        """
        ## Configuration.
        self.phi = phi
        ## Selected trajectory point.
        self.trajPoint = trajPoint
        ## dict(str -> complex) of log of all weights, always contains key "actVal".
        self.logWeights = logWeights if logWeights is not None else dict()
        self.logWeights["actVal"] = actVal
        ## dict(str -> object) of extra data.
        self.extra = extra if extra is not None else dict()

    @property
    def actVal(self):
        """!
        Value of the action at phi (complex).
        """
        return self.logWeights["actVal"]

    @actVal.setter
    def actVal(self, actVal):
        self.logWeights["actVal"] = actVal

    def nonActWeigths(self):
        """!
        Iterator over key, value pairs of all log weights other than the action.
        """
        yield from filter(lambda item: item[0] != "actVal",
                          self.logWeights.items())

    def accept(self, phi, actVal, logWeights=None, extra=None):
        """!
        Return a new EvolutionStage which indicates acceptance with given parameters.
        """
        return self.__class__(phi, actVal, 1, logWeights, extra)

    def reject(self, extra=None):
        """!
        Return a new EvolutionStage which indicates rejection and reuses phi and actVal.
        `logWeights` and `extra` are always overwritten.
        """
        return self.__class__(self.phi, self.actVal, 0, self.logWeights, extra)

    def sumLogWeights(self):
        """!
        Return the sum of all log weights.
        """
        return sum(self.logWeights.values())

    def save(self, h5group):
        """!
        Save contents to an HDF5 group.
        """
        h5group["phi"] = self.phi
        h5group["actVal"] = self.actVal
        h5group["trajPoint"] = self.trajPoint

        # write additional weights is there are any
        if len(self.logWeights) > 1:
            wgrp = h5group.create_group("logWeights")
            for key, val in self.nonActWeigths():
                wgrp[key] = complex(val)

        if self.extra:
            egrp = h5group.create_group("extra")
            for key, val in self.extra():
                egrp[key] = val

    @classmethod
    def fromH5(cls, h5group):
        """!
        Construct a new instance from an HDF5 group written by EvolutionStage.save().
        """
        extra = {key: dset[()] for key, dset in h5group["extra"].items()} \
            if "extra" in h5group else None
        logWeights = {key: dset[()] for key, dset in h5group["logWeights"].items()} \
            if "logWeights" in h5group else None
        return cls(Vector(h5group["phi"][()]),
                   h5group["actVal"][()],
                   h5group["trajPoint"][()],
                   logWeights,
                   extra)
