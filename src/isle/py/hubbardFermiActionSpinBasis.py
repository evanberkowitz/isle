import isle

# TODO adjust determinant in measurements!

class HubbardFermiActionSpinBasis(isle.Action):
    def __init__(self, kappa, mu, sigmaKappa):
        # TODO can we call super ctor?
        isle.Action.__init__(self)
        self._hfa = isle.HubbardFermiAction(kappa, mu, sigmaKappa)

    def eval(self, phi):
        return self._hfa.eval(-1j*phi)

    def force(self, phi):
        return -1j*self._hfa.force(-1j*phi)
