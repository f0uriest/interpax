"""Util functions for interpax."""

import functools
import warnings
from typing import Any, Type, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, Num
from numpy.typing import ArrayLike

# jax.typing.ArrayLike and jaxtyping.ArrayLike don't include eg tuples,lists,iterables
# like np.ArrayLike. This combines all the usual array types
Arrayish = Union[Array, ArrayLike]


def isbool(x: Any) -> bool:
    """Check if something is boolean or ndarray of bool type."""
    return isinstance(x, bool) or (hasattr(x, "dtype") and (x.dtype == bool))


def errorif(
    cond: Union[bool, jax.Array], err: Type[Exception] = ValueError, msg: str = ""
):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


def warnif(
    cond: Union[bool, jax.Array], err: Type[Warning] = UserWarning, msg: str = ""
):
    """Throw a warning if condition is met."""
    if cond:
        warnings.warn(msg, err)


def asarray_inexact(x: Num[Arrayish, "..."]) -> Inexact[Array, "..."]:
    """Convert to jax array with floating point dtype."""
    x = jnp.asarray(x)
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(x, jnp.array(1.0))
    return x.astype(dtype)


def wrap_jit(*args, **kwargs):
    """Wrap a function with jit with optional extra args.

    This is a helper to ensure docstrings and type hints are correctly propagated
    to the wrapped function, bc vscode seems to have issues with regular jitted funcs.
    """

    def wrapper(fun):
        foo = jax.jit(fun, *args, **kwargs)
        foo = functools.wraps(fun)(foo)
        return foo

    return wrapper


def safediv(a, b, fill=0.0, threshold=0.0):
    """Divide a/b with guards for division by zero.

    Parameters
    ----------
    a, b : ndarray
        Numerator and denominator.
    fill : float, ndarray, optional
        Value to return where b is zero.
    threshold : float >= 0
        How small is b allowed to be.

    """
    mask = jnp.abs(b) <= threshold
    num = jnp.where(mask, fill, a)
    den = jnp.where(mask, 1.0, b)
    return num / den
