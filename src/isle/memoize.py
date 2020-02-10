r"""!\file
\brief Utilities for memoizing function results.

Memoization decorators let us compute expensive functions once and store the result,
so that if the function is called again with the same arguments the result is just returned.
"""


import functools
import inspect
import typing
import weakref
from dataclasses import dataclass
from logging import getLogger


class MemoizeMethod:
    r"""!
    Decorator that memoizes the result of a method call based on given arguments.

    \warning This decorator works only on bound methods not free functions.

    \note Classes using this decorator on their methods must support hashing.

    The most recent result of calling the decorated method is cached and returned on subsequent
    calls if the specified arguments match those of the cached call.
    This can help avoid repeating expensive calculations in a way that is transparent to the user.
    It is important that all relevant arguments are checked though, so as not to re-use results erroneously.

    <B>Example: Memoize by one argument</B><br>
    ```{.py}
    class C:
        @MemoizeMethod("a")
        def f(self, a, b):
            print(f"evaluate f({a}, {b})")

    c1 = C()
    c1.f(0, 1)  # evaluated
    c1.f(0, 1)  # re-used
    c1.f(0, 2)  # re-used; careful, this might be wrong!
    c1.f(1, 1)  # evaluated, a has changed
    c1.f(1, 1)  # re-used
    c1.f(0, 1)  # evaluated, only the most recent result is stored

    c2 = C()
    c2.f(0, 1)  # evaluated, c1 and c2 do not share memoized results
    c1.f(0, 1)  # re-used from earlier call to c1.f
    ```

    <B>Example: Memoize by two arguments</B><br>
    ```{.py}
    class C:
        @MemoizeMethod("a", "b")
        def f(self, a, b):
            print(f"evaluate f({a}, {b})")

    c1 = C()
    c1.f(0, 1)  # evaluated
    c1.f(0, 1)  # re-used
    c1.f(0, 2)  # evaluated (b changed)
    c1.f(1, 2)  # evaluated (a changed)
    c1.f(1, 2)  # re-used
    c1.f(0, 2)  # evaluated, only the most recent result is stored
    ```

    \note The efficiency of this decorator depends on the equality operator (`==`).
          If it is slow for the arguments by which the result is memoized, function calls
          can still be expensive.
    """

    #
    # *** Implementation notes ***
    #
    # Using the decorator syntax like in the examples applies the MemoizeMethod decorator to
    # the function at *class* creation time, i.e. before it is bound to an instance.
    # This means that all instances of a class share the same instance of MemoizedMethod
    # for each of their decorated methods.
    # It is thus necessary to distinguish the different instances and store arguments
    # and return values for each one separately.
    #
    # This is achieved by keeping weak references to all instances on which the
    # decorated method get called in the attribute _instanceData.
    # This happens when the method gets called and not when the instance is created.
    #

    @dataclass
    class MemoizedData:
        """!
        Store values of arguments and the return value of the memoized function.
        """
        # Stored values of arguments as processed by argumentKey.
        # Can be `_Empty`, meaning that the method has not been called yet.
        argvals: typing.Any
        # Return value.
        result: typing.Any

    def __init__(self, argumentKeyFn):
        r"""!
        \param argumentKeyFn Callable which is applied to the arguments of the decorated method
                             (except for `self`) before comparing to the cache.
        """
        self.argumentKeyFn = argumentKeyFn
        self._argumentKeyFnParams = inspect.signature(self.argumentKeyFn).parameters
        self._instanceData = weakref.WeakKeyDictionary()

    def __call__(self, method):
        """!
        Apply decorator to a method.
        """

        memo = self  # Using 'self' inside of wrapper is confusing; 'memo' is the instance of Memoize.
        methodSignature = self._getMethodSignature(method)

        @functools.wraps(method)
        def wrapper(instance, *args, **kwargs):
            memoized = memo._getOrInsertInstanceData(instance)
            actualArguments = _bindArguments(methodSignature, instance, *args, **kwargs)
            argumentKey = memo.argumentKeyFn(**memo._keyArguments(actualArguments))

            if memoized.argvals is not _Empty:  # Has the function been called before?
                if memoized.argvals == argumentKey:
                    # The previous call was equivalent to the current one.
                    return memoized.result

            memoized.argvals = argumentKey
            memoized.result = method(instance, *args, **kwargs)
            return memoized.result

        return wrapper

    def _keyArguments(self, allArguments):
        """!
        Return the arguments for the key functions based on all arguments passed to method.
        """
        return {name: allArguments[name] for name in self._argumentKeyFnParams}

    def _getMethodSignature(self, method):
        """!
        Return and verify the signature of `method`.
        """

        methodSig = inspect.signature(method)
        for name in self._argumentKeyFnParams:
            if name not in methodSig.parameters:
                getLogger(__name__).error("Argument name '%s' used in key function not in signature"
                                          " of function %s\n  signature: %s",
                                          name, method, methodSig)
                raise TypeError(f"Argument name {name} if not part of function signature")
        return methodSig

    def _getOrInsertInstanceData(self, instance):
        """!
        Return the MemoizationData object for given instance.
        Creates a new one if `instance` has not been seen before.
        """

        try:
            return self._instanceData[instance]
        except KeyError:
            getLogger(__name__).debug("Inserting new instance for memoization: %s\n  in memoization object %s",
                                      instance, self)
            self._instanceData[instance] = self.MemoizedData(_Empty, None)
            return self._instanceData[instance]


def _bindArguments(signature, *args, **kwargs):
    r"""!
    Bind the given arguments including default values to a function signature and return
    an ordered mapping from argument names to values.
    """
    boundArguments = signature.bind(*args, **kwargs)
    boundArguments.apply_defaults()
    return boundArguments.arguments


class _EmptyType:
    """!
    Indicates that some variable has not been set yet.
    """

    def __repr__(self):
        return "Empty"

    def __bool__(self):
        return False

    def __reduce__(self):
        return "Empty"


_Empty = _EmptyType()
