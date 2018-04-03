"""!
Some general utilities.
"""

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
