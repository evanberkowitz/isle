r"""!\file
Utilities for memoization.

Memoization decorators let us compute expensive functions once and store the result,
so that if the function is called again with the same arguments the result is just returned.
"""


import inspect
import typing
import weakref
from dataclasses import dataclass
from logging import getLogger


class MemoizeMethod:
    r"""!
    Memoize the result of a function call based on given arguments.

    <B>Example</B><br>
    ```{.py}
    # Memoize by argument a
    @one_value_by("b")
    def f(a, b, foo):
        print("evaluated f")
        return a+1, b-1, foo+foo

    print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated)")
    print("f(1,1,'bar') -->", f(1,1,"bar"), "   (re-used)")
    print("f(1,2,'bar') -->", f(1,2,"bar"), "   (evaluated)")
    print("f(1,2,'bar') -->", f(1,2,"bar"), "   (re-used)")
    print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated, even though it was recently evaluated)")
    print("f(2,1,'bar') -->", f(2,1,"bar"), "   (re-used, even though a changed, because memoization was by b)")

    # Memoize by a and b
    @one_value_by("b", "a")
    def f(a, b, foo):
        print("evaluated f")
        return a+1, b-1, foo+foo

    print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated)")
    print("f(1,1,'bar') -->", f(1,1,"bar"), "   (re-used)")
    print("f(1,2,'bar') -->", f(1,2,"bar"), "   (evaluated)")
    print("f(1,2,'bar') -->", f(1,2,"bar"), "   (re-used)")
    print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated, even though it was recently evaluated)")
    print("f(2,1,'bar') -->", f(2,1,"bar"), "   (evaluated, unlike before, because we care about a now)")
    ```

    \note The efficiency of this decorator depends on the equality operator (`==`).
          If it is slow for the arguments by which the result is memoized, function calls
          can still be expensive.
    """


    @dataclass
    class MemoizedData:
        argvals: typing.Dict[str, typing.Any]
        result: typing.Any

    def __init__(self, *argnames):
        self.argnames = argnames
        self._instanceData = weakref.WeakKeyDictionary()

    def __call__(self, function):
        memo = self  # Using 'self' inside of wrapper is confusing; 'memo' is the instance of Memoize.
        signature = inspect.signature(function)

        def wrapper(instance, *args, **kwargs):
            memoized = memo._getOrInsertInstanceData(instance)
            actualArguments = _bindArguments(signature, instance, *args, **kwargs)

            if memoized.argvals is not None:
                if all(memoized.argvals[argname] == actualArguments[argname] for argname in memo.argnames):
                    return memoized.result

            memoized.argvals = {argname: actualArguments[argname] for argname in memo.argnames}
            memoized.result = function(instance, *args, **kwargs)
            return memoized.result

        return wrapper

    def _getOrInsertInstanceData(self, instance):
        try:
            return self._instanceData[instance]
        except KeyError:
            getLogger(__name__).debug("Inserting new instance for memoization: %s\n in memoization object %s",
                                      instance, self)
            self._instanceData[instance] = self.MemoizedData(None, None)
            return self._instanceData[instance]


def _bindArguments(signature, *args, **kwargs):
    r"""!
    Bind the given arguments including default values to a function signature and return
    an ordered mapping from argument names to values.
    """
    boundArguments = signature.bind(*args, **kwargs)
    boundArguments.apply_defaults()
    return boundArguments.arguments
