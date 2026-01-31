# exception.py - exception definitions for the control package
#
# Initial author: Richard M. Murray
# Creation date: 31 May 2010

"""Exception definitions for the control package."""

import warnings


class ControlSlicot(ImportError):
    """Slicot import failed."""
    pass


def _deprecated_alias(old_name, new_name):
    """Issue deprecation warning for renamed class/function."""
    warnings.warn(
        f"{old_name} is deprecated, use {new_name} instead",
        DeprecationWarning, stacklevel=3
    )


class ControlSlycot(ControlSlicot):
    """Deprecated alias for ControlSlicot."""
    def __init__(self, *args, **kwargs):
        _deprecated_alias('ControlSlycot', 'ControlSlicot')
        super().__init__(*args, **kwargs)

class ControlDimension(ValueError):
    """Raised when dimensions of system objects are not correct."""
    pass

class ControlArgument(TypeError):
    """Raised when arguments to a function are not correct."""
    pass

class ControlIndexError(IndexError):
    """Raised when arguments to an indexed object are not correct."""
    pass

class ControlMIMONotImplemented(NotImplementedError):
    """Function is not currently implemented for MIMO systems."""
    pass

class ControlNotImplemented(NotImplementedError):
    """Functionality is not yet implemented."""
    pass

# Utility function to see if slicot is installed
slicot_installed = None
def slicot_check():
    """Return True if slicot is installed, otherwise False."""
    global slicot_installed
    if slicot_installed is None:
        try:
            import slicot  # noqa: F401
            slicot_installed = True
        except:
            slicot_installed = False
    return slicot_installed


# Backwards-compatible alias (no warning to avoid noise in existing code)
slycot_check = slicot_check


# Utility function to see if pandas is installed
pandas_installed = None
def pandas_check():
    """Return True if pandas is installed, otherwise False."""
    global pandas_installed
    if pandas_installed is None:
        try:
            import pandas  # noqa: F401
            pandas_installed = True
        except:
            pandas_installed = False
    return pandas_installed

# Utility function to see if cvxopt is installed
cvxopt_installed = None
def cvxopt_check():
    """Return True if cvxopt is installed, otherwise False."""
    global cvxopt_installed
    if cvxopt_installed is None:
        try:
            import cvxopt  # noqa: F401
            cvxopt_installed = True
        except:
            cvxopt_installed = False
    return cvxopt_installed
