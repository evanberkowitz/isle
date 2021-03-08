from .configure import configure_command
from .cmake_extension import CMakeExtension
from .build import get_cmake_builder
from .version import version_from_git
from .docs import doxygen_command
from .tests import TestCommand
from . import predicate
