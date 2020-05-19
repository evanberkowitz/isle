r"""!\file
\ingroup evolvers
Evolvers that use symmetries of the Hubbard action to perform large jumps in configuration space.
"""

from logging import getLogger

from .evolver import Evolver


class Alternator(Evolver):
    r"""! \ingroup evolvers
    Alternate between different evolvers.

    Each sub evolver is run a given number of times.
    Then Alternator advances to the next evolver and runs it for its
    number of iterations.
    After the last sub evolver, Alternator cycles round to the first one and repeats.

    <B>Example</B><br>
    This example defines an %Alternator which uses ConstStepLeapfrog for 100 trajectories,
    then TwoPiJumps once, and repeats.
    ```{.py}
    # Set up parameters
    # hmcState, length, nstep
    # ...

    evolver = isle.evolver.Alternator((
        (100, isle.evolver.ConstStepLeapfrog(hmcState.action, length, nstep, hmcState.rng)),
        (1, isle.evolver.hubbard.TwoPiJumps(hmcState.lattice.lattSize(), 1, hmcState.action,
                                            hmcState.lattice, hmcState.rng))
    ))
    ```
    """

    def __init__(self, evolvers=None, startIndex=0):
        r"""!
        Store given sub evolvers.
        \param evolvers List of lists of the form `[[c0, e0], [c1, e1], ...]`,
                        where `ei` are the sub evolvers to cycle through.
                        `ci` is the number of iterations to run `ei` for.
        \param startIndex Start iteration at this point. Exists mostly for internal use.
        """
        if evolvers:
            self._counts, self._subEvolvers = map(list, zip(*evolvers))
        else:
            self._subEvolvers = []
            self._counts = []

        self._current = startIndex

    def add(self, count, evolver):
        r"""!
        Add a sub evolver to the Alternator.
        \param count Number of iterations the evolver shall be run for.
        \param evolver Sub evolver to add.
        """

        if count < 1:
            getLogger(__name__).warning(
                "Iteration count for evolver %s is less than one: %d",
                evolver, count)

        self._counts.append(count)
        self._subEvolvers.append(evolver)

    def _advance(self):
        """!Advance internal evolver index."""
        self._current = (self._current + 1) % sum(self._counts)

    def _pickCurrentEvolver(self):
        """!Pick a sub evolver based on the current evolver index."""
        currentTotalCount = 0
        for count, evolver in zip(self._counts, self._subEvolvers):
            currentTotalCount += count
            if currentTotalCount > self._current:
                return evolver
        return None

    def evolve(self, stage):
        r"""!
        Delegate to a sub evolver next in line.
        \param stage EvolutionStage at the beginning of this evolution step.
        \returns EvolutionStage at the end of this evolution step.
        """

        subEvolver = self._pickCurrentEvolver()
        self._advance()

        return subEvolver.evolve(stage)

    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """
        h5group["counts"] = self._counts
        h5group["current"] = self._current
        for idx, evolver in enumerate(self._subEvolvers):
            grp = h5group.create_group(f"sub{idx}")
            manager.save(evolver, grp)

    @classmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed alternating evolver.
        """
        alternator = cls(startIndex=h5group["current"][()])
        counts = h5group["counts"][()]
        for idx, count in enumerate(counts):
            subEvolver = manager.load(h5group[f"sub{idx}"], action, lattice, rng)
            alternator.add(count, subEvolver)
        return alternator

    def report(self):
        r"""!
        Return a string summarizing the evolution since the evolver
        was constructed including by fromH5.
        """
        reports = "\n".join(f"  {i}. frequency = {count}\n"+
                            "\n".join("    "+line for line in evolver.report().split("\n"))
                            for i, (count, evolver) in enumerate(zip(self._counts, self._subEvolvers)))
        return f"<Alternator> (0x{id(self):x})\n{reports}"
