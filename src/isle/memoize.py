"""!
Memoization decorators let us compute expensive functions once and store the result, so that if the function is called again with the same arguments the result is just returned.
"""

class one_value_by():
    def __init__(self, *variables_as_strings):
        self.variables = variables_as_strings       # Names of variables.

        if self.variables == ():
            raise ValueError("You must specify at least one argument by which to memoize.")

        self.indices = list()                       # Where they are in the function definition.
        self.recent  = None                         # Recent values of the arguments to check against.
        self.value   = None                         # Cached value so we don't evaluate an expensive function too much
    
    def __call__(self, function):
        
        self.indices = [function.__code__.co_varnames.index(v) for v in self.variables]
        
        def wrapper(*arguments):
            if self.recent:                                         # Check that we've evaluated at all in the past.
                        
                for i in self.indices:                              # Use the obscure for-else python construction:
                    if self.recent[i] != arguments[i]:              # If all the self.recent match the arguments,
                        break                                       # there won't be a break,
                else:                                               # We'll exit the for loop normally
                    # print("Using memoized value!")
                    return self.value                               # and we can use the memoized value.
            
                                                                    # If an argument differs from the recent evaluation, (or it's our first evaluation)
            self.recent = {i: arguments[i] for i in self.indices}   # We store the arguments to compare with next time
            self.value  = function(*arguments)                      # and evaluate the function and store the result,
            return self.value                                       # returning the new value.
        
        return wrapper


print("\n\nHere's a simple demo usage of the @one_value_by memoization decorator on f(a,b,foo):")

@one_value_by("b")         # Works
def f(a, b, foo):
    print("f is expensive and was evaluated!")
    return a+1, b-1, foo+foo

print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated)")
print("f(1,1,'bar') -->", f(1,1,"bar"), "   (re-used)")
print("f(1,2,'bar') -->", f(1,2,"bar"), "   (evaluated)")
print("f(1,2,'bar') -->", f(1,2,"bar"), "   (re-used)")
print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated, even though it was recently evaluated)")
print("f(2,1,'bar') -->", f(2,1,"bar"), "   (re-used, even though a changed, because memoization was by b!)")

print("\n\nHere's the same demo, but now consider both b and a for memoization of f(a,b,foo):")
@one_value_by("b", "a")     # Works
def f(a, b, foo):
    print("f is expensive and was evaluated!")
    return a+1, b-1, foo+foo

print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated)")
print("f(1,1,'bar') -->", f(1,1,"bar"), "   (re-used)")
print("f(1,2,'bar') -->", f(1,2,"bar"), "   (evaluated)")
print("f(1,2,'bar') -->", f(1,2,"bar"), "   (re-used)")
print("f(1,1,'bar') -->", f(1,1,"bar"), "   (evaluated, even though it was recently evaluated)")
print("f(2,1,'bar') -->", f(2,1,"bar"), "   (evaluated, unlike before, because we care about a now)")
