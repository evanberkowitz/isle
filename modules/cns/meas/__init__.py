"""!
General measurements that can be useful in various contexts.

Canonically, a measurement is a class witha name in UpperCamelCase in a module
of the same name but in lowerCamelCase. Such a class is imported into the <TT>cns.meas</TT>
namespace. In other cases, the measurements have to be addressed by using the module name
in addition to the class or function name.

More details can be found under \ref measdoc "Measurements".
"""

## \defgroup meas Measurements
# Perform measurements on configurations.

## \cond DO_NOT_DOCUMENT

# do not import these modules
IGNORED = ("common")

# function to hide variables
def _importAll():
    from importlib import import_module
    from pathlib import Path
    measPath = Path(__file__).resolve().parent  # path to meas module

    # look at all files in directory
    for filename in Path(measPath).iterdir():
        if filename.suffix != ".py":
            continue # not a module

        modname = filename.stem  # name without path and suffix
        if not modname.startswith("__") and modname not in IGNORED:
            # import the module and its main class
            classname = modname[0].capitalize()+modname[1:] # capitalize first letter
            mod = import_module("."+modname, "cns.meas")
            try:
                globals()[classname] = mod.__dict__[classname]
            except KeyError:
                pass # do not import class

_importAll()
del _importAll
del IGNORED

## \endcond
