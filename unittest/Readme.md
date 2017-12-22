# Usage
- Copy the compiled `cns` python library to this directory for running unittests.
- Run `python example.py` or `python example.py -v` where `python` can import cns.
# Misc
- See https://docs.python.org/3/library/unittest.html for more examples.
- One can express the operators `__add__`, `__iadd__`, ..  as functions of two vectors using the `import operator` module: `operator.iadd(vec1, vec2) <-> vec1+=vec2`. Thus one can easily iterate over all vector operations.