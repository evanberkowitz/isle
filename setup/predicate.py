"""
Unary functions that check if a certain condition holds for the argument.

All predicate functions print a message to STDOUT if they are not fulfilled
and return False.
"""

import subprocess
from pathlib import Path

def executable(cmd):
    "Check if cmd is an executable command."
    status, output = subprocess.getstatusoutput(f"which {cmd}")
    if status == 0:
        return True
    print(f"error: Cannot execute command {cmd}:")
    print(f"  {output}")
    return False

def file(fname):
    "Check if fname is a file."
    path = Path(fname)
    if not path.exists():
        print(f"error: File does not exist: {fname}")
        return False
    if not path.is_file():
        print(f"error: Not a file: {fname}")
        return False
    return True

def directory(dname):
    "Check if dname is a directory."
    path = Path(dname)
    if not path.exists():
        print(f"error: File does not exist: {dname}")
        return False
    if not path.is_dir():
        print(f"error: Not a directory: {dname}")
        return False
    return True

def one_of(*allowed):
    "Return a function that checks whether a value is in allowed."
    def _one_of_impl(val):
        if val not in allowed:
            print(f"error: Value not allowed: '{val}', must be one of {allowed}")
            return False
        return True
    return _one_of_impl
