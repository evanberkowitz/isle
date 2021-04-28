"""
Versioning utilities.
"""

import subprocess
import re

def version_from_git(plain=True):
    """
    Get a version number from the latest git tag.
    Tags have to contain a version number of the form
    v0.1, v0.1.3, etc. where the leading v is optional.
    The number may be followed by arbitrary text.

    If plain=True, only the version number is returned, otherwise anything following
    the number is included as well. The v is always omitted.
    """

    try:
        latest_tag = subprocess.check_output(["git", "describe", "--tag"],
                                             stderr=subprocess.STDOUT).decode("utf-8")
        match = re.match(r"v?((\d[\d\.]*).*)\n?", str(latest_tag))
        if match is None:
            print(f"error: latest git tag does not conform to format of version number: {latest_tag}")
            return "0.0-unknown"
        if plain:
            return match[2]
        return match[1]
    except subprocess.CalledProcessError:
        return "0.0-unknown"
