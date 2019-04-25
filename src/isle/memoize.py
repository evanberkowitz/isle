r"""!\file
Utilities for memoization.

Memoization decorators let us compute expensive functions once and store the result,
so that if the function is called again with the same arguments the result is just returned.
"""

class one_value_by():
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

    def __init__(self, *variables_as_strings):
        ## Names of variables.
        self.variables = variables_as_strings

        if not self.variables:
            raise ValueError("You must specify at least one argument by which to memoize.")

        ## Where the arguments are in the function definition.
        self.indices = list()
        ## Recent values of the arguments to check against.
        self.recent = None
        ## Cached value so we don't evaluate an expensive function too much
        self.value = None

    def __call__(self, function):
        """!Apply the decorator to a function."""

        self.indices = [function.__code__.co_varnames.index(v) for v in self.variables]

        def wrapper(*arguments):
            if self.recent is not None:  # Check that we've evaluated at all in the past.

                # If all the self.recent match the arguments, we can use the memoized value.
                if all(self.recent[i] == arguments[i] for i in self.indices):
                    return self.value

            # If an argument differs from the recent evaluation, (or it's our first evaluation)
            self.recent = {i: arguments[i] for i in self.indices}   # We store the arguments to compare with next time
            self.value = function(*arguments)                       # and evaluate the function and store the result,
            return self.value                                       # returning the new value.

        return wrapper
