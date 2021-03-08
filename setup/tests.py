"""
Run tests using pytest
"""

import sys
import subprocess
import distutils.cmd

class TestCommand(distutils.cmd.Command):
    description = "run tests"
    user_options = [("select=", "k", """only run tests which match the given substring expression. An expression is a python
evaluatable expression where all names are substring-matched against test names and their
parent classes. Example: -k 'test_method or test_other' matches all test functions and
classes whose name contains 'test_method' or 'test_other', while -k 'not test_method'
matches those that don't contain 'test_method' in their names. -k 'not test_method and not
test_other' will eliminate the matches. Additionally keywords are matched to classes and
functions containing extra names in their 'extra_keyword_matches' set, as well as functions
which have names assigned directly to them. The matching is case-insensitive.""")]

    def initialize_options(self):
        self.select = None

    def finalize_options(self):
        pass

    def run(self):
        "Execute the command."

        try:
            subprocess.check_call(["pytest"] + (["-k", self.select] if self.select else []),
                                  cwd="tests")
        except subprocess.CalledProcessError as exc:
            print(f"error: Could not run pytest: {exc}")
            sys.exit(1)
