"""!\file
Various helpers for meta programming in Python.
"""

import inspect
from logging import getLogger


def sourceof(typ):
    """!
    Return the source code of a type definition with indentation
    according to the first line stripped off.
    """

    lines = inspect.getsource(typ).split("\n")
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)

def sourceOfFunction(func):
    r"""!
    Return the source code of a function.
    Works only on free functions, not methods or lambdas.

    \see functionFromSource() for the inverse.
    """

    if not inspect.isfunction(func) or inspect.ismethod(func):
        raise ValueError("Not a function, bound methods not allowed")

    src = sourceof(func)

    # lambdas are weird, need do make sure this is a normal function definition
    if not src.lstrip().startswith("def"):
        raise ValueError("Not a proper function, lambdas not allowed")

    return src

def sourceOfClass(cls):
    r"""!
    Return the source code of a class.

    \see classFromSource() for the inverse.
    """

    # make sure we have the type not an instance
    if not isinstance(cls, type):
        cls = type(cls)

    if not inspect.isclass(cls):
        raise ValueError("Not a class")

    return sourceof(cls)


def defineFromSource(src, definitions={}):
    r"""!
    Extract and execute a definition from a piece of source code.

    \param src Source code.
    \param definitions Dict mapping names to types. Use this to provide
                       objects that can be referenced by the source code.
    """

    # an empty local scope to store new class definition
    scope = dict()
    exec(src, definitions, scope)

    if not scope:
        getLogger(__name__).error("Tried to extract a definition from source code but "
                                  "the source code does not define anything. Source:\n%s",
                                  src)
        raise ValueError("Nothing defined by source code")
    if len(scope) > 1:
        getLogger(__name__).error("Source code define more than one thing. Cannot extract "
                                  "object unambiguously. Source:\n%s",
                                  src)
        raise ValueError("More than one thing defined")

    # the only thing in scope is the one we want
    return next(iter(scope.values()))

def functionFromSource(src, definitions={}):
    r"""!
    Return a function defined by a piece of source code.

    \see sourceOfFunction() for the inverse.
    """

    obj = defineFromSource(src, definitions)

    if not inspect.isfunction(obj):
        getLogger(__name__).error("Expected a function definition in source code "
                                  "but got something else. Source:\n%s",
                                  src)
        raise ValueError("Source code defined something else than a function")

    return obj

def callFunctionFromSource(src, *args, **kwargs):
    r"""!
    Extract a function from source code and call it.

    `args` and `kwargs` are passed on to the function and its result is returned.

    \see sourceOfFunction() and functionFromSource().
    """

    func = functionFromSource(src)
    try:
        return func(*args, **kwargs)
    except NameError as e:
        getLogger(__name__).error("Undefined symbol in function constructed from "
                                  "source: %s; source code:\n%s",
                                  str(e), src)
        raise
    except:
        # must re-raise it so things like keyboard interrupts get processed properly
        raise

def classFromSource(src, definitions={}):
    r"""!
    Return a class defined by a piece of source code.

    \see sourceOfClass() for the inverse.
    """

    obj = defineFromSource(src, definitions)

    if not inspect.isclass(obj):
        getLogger(__name__).error("Expected a class definition in source code "
                                  "but got something else. Source:\n%s",
                                  src)
        raise ValueError("Source code defined something else than a class")

    return obj
