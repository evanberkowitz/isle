"""!
Simply report the progress.
"""

import numpy as np

class Progress:
    r"""!
    \ingroup meas
    Print a simple report on the progress of the calculation.
    """

    def __init__(self, stepName, targetIteration, startIteration=0):
        r"""!
        Create a Progress report.
        \param stepName A string that indicates what kind of progress is being made.
        \param targetIteration When the iteration count reaches `targetIteration`
               the computation should be complete.
        \param startIteration  Sets 0% complete to be at `startIteration`
        """
        self.name = stepName
        self.start = startIteration
        self.current = self.start
        self.percent = 0
        self.target = targetIteration
        digits = str(int(np.ceil(np.log10(self.target))))
        self.string = "{} {:5.2f}% complete at iteration {:"+digits+"} (aiming for {:"+digits+"})."

    def __call__(self, phi, inline=True, **kwargs):
        """!Print progress report."""
        if inline:
            self.current = kwargs['itr']
            self.percent = 100 * (self.current-self.start) / (self.target-self.start)
            print(self.string.format(self.name, self.percent, self.current, self.target))
        else:
            pass

    def report(self):
        """!Print most recent progress report."""
        print(self.string.format(self.name, self.percent, self.current, self.target))
