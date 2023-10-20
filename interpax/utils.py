"""Util functions for interpax."""

import warnings


def isbool(x):
    """Check if something is boolean or ndarray of bool type."""
    return isinstance(x, bool) or (hasattr(x, "dtype") and (x.dtype == bool))


def errorif(cond, err=ValueError, msg=""):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


def warnif(cond, err=UserWarning, msg=""):
    """Throw a warning if condition is met."""
    if cond:
        warnings.warn(msg, err)
