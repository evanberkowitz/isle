r"""!
Handle collections of elements.
"""

from logging import getLogger
import math


def hingeRange(start, end, stepSize):
    r"""!
    A generator that behaves similarly to the builtin range with two differences:
    `%hingeRange`
      - keeps going after reaching the end value and yields `end` forever and
      - accepts floats as parameters.

    \param start Start value (inclusive).
    \param end End value (exculsive if called only `int((end-start)/stepSize)` times).
    \param stepSize Size of the steps while iterating from start to end.
                    Must be negative if `end < start`.
    """

    cur = start  # current value
    # iterate from start to end
    while (stepSize > 0 and cur < end) or (stepSize < 0 and cur > end):
        yield cur
        cur += stepSize
    # stay at end value forever
    while True:
        yield end

def listToSlice(lst):
    r"""!
    Convert a list to a `slice`.
    \param lst List of values in a uniform range.
    \returns `slice s` such that `list(range(s.start, s.stop, s.step)) == lst`.
    \throws ValueError if list has less than two elements or the step size is not uniform.
    """

    if len(lst) < 2:
        raise ValueError("List must have at least two entries to convert to slice")

    diffs = [l - lst[i-1] for i, l in enumerate(lst[1:], 1)]
    if diffs.count(diffs[0]) != len(diffs):
        raise ValueError("Not all step sizes are equal in list, can't convert to slice")

    return slice(lst[0], lst[-1]+diffs[0], diffs[0])

def parseSlice(string, minComponents=0, maxComponents=3):
    r"""!
    Parse a string as a slice in usual :-notation.
    Allows for components to be `None` or `"none"` (case insensitive).
    \param string Input string.
    \param minComponents Minimum number of slice components required in string.
    \param minComponents Maxiumum number of slice components allowed in string.
    \return `slice` object constructed from `string`.
    """

    components = list(map(lambda x: None if x is None or x.strip().lower() in ("", "none") \
                          else int(x),
                          string.split(":")))
    if len(components) < minComponents:
        raise ValueError(f"Too few components in string {string} for parsing as slice, "
                         f"requires at least {minComponents}")
    if len(components) > maxComponents:
        raise ValueError(f"Too many components in string {string} for parsing as slice, "
                         f"requires at most {maxComponents}")

    return slice(*components)


def _isValidSubslice(large, small):
    """!
    Check if a small slice allows constructing an array from data in a larger array.
    """

    log = getLogger(__name__)

    if any(x is None for x in (large.start, large.stop, large.step,
                               small.start, small.stop, small.step)):
        log.error("All slice parameters must be given, None is not allowed")
        return False
    if any(x < 0 for x in (large.start, large.stop, large.step,
                           small.start, small.stop, small.step)):
        log.error("No slice parameters may be less than 0")
        return False

    if small.step % large.step != 0:
        log.error("Step of small slice (%d) must be a multiple of step of large slice (%d)",
                  small.step, large.step)
        return False

    if small.start < large.start:
        log.error("Small start (%d) must not be less than large start (%d)",
                  small.start, large.start)
        return False
    if (small.start - large.start) % large.step != 0:
        log.error("Small start (%d) must be reachable from large start (%d) "
                  "in steps of size large.step (%d)",
                  small.start, large.start, large.step)
        return False

    if small.stop > large.stop:
        log.error("Small start (%d) must not be greater than large stop (%d)",
                  small.stop, large.stop)
        return False
    # Do not need any extra check for small.stop, it does not need to be reachable from
    # either large.start or large.stop.

    if small.stop < small.start:
        log.error("Small start (%d) must not be greater than small stop (%d)",
                  small.start, small.stop)
        return False
    if large.stop < large.start:
        log.error("Large start (%d) must not be greater than large stop (%d)",
                  large.start, large.stop)
        return False

    return True

def subslice(large, small):
    if not _isValidSubslice(large, small):
        raise ValueError(f"Invalid slices: large={large}, small={small}")

    step = small.step // large.step
    start = (small.start - large.start) // large.step
    stop = start + math.ceil((small.stop - small.start)/small.step)*step
    return slice(start, stop, step)