"""
Extension for setuptools that uses CMake.
"""

from pathlib import Path
import setuptools

class CMakeExtension(setuptools.Extension):
    "Represents an extension built using CMake"
    def __init__(self, name, sourcedir=""):
        """
        Arguments:
           - name: Name of the extension
           - sourcedir: Directory of the root CMakeLists.txt
        """
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()
