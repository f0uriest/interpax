"""Util functions for interpax."""

import warnings

import jax.numpy as jnp


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


def asarray_inexact(x):
    """Convert to jax array with floating point dtype."""
    x = jnp.asarray(x)
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(x, jnp.array(1.0))
    return x.astype(dtype)
