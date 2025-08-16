from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Inexact, Num

from .utils import asarray_inexact, wrap_jit


@wrap_jit(static_argnames=["n"])
def fft_interp1d(
    f: Num[ArrayLike, "nx ..."],
    n: int,
    sx: Optional[Num[ArrayLike, " s"]] = None,
    dx: float = 1.0,
) -> Inexact[Array, "n ... s"]:
    """Interpolation of a real-valued 1D periodic function via FFT.

    Parameters
    ----------
    f : ndarray, shape(nx, ...)
        Source data. Assumed to cover 1 full period, excluding the endpoint.
    n : int
        Number of desired interpolation points.
    sx : ndarray or None
        Shift in x to evaluate at. If original data is f(x), interpolates to f(x + sx)
    dx : float
        Spacing of source points

    Returns
    -------
    fi : ndarray, shape(n, ..., len(sx))
        Interpolated (and possibly shifted) data points
    """
    f = asarray_inexact(f)
    nx = f.shape[0]
    c = jnp.fft.rfft(f, axis=0, norm="forward")

    if sx is not None:
        tau = 2 * jnp.pi
        sx = asarray_inexact(sx)
        sx = jnp.exp(1j * jnp.fft.rfftfreq(nx, dx / tau)[:, None] * sx)
        c = (c[None].T * sx).T
        c = jnp.moveaxis(c, 0, -1)

    return jnp.fft.irfft(c, n, axis=0, norm="forward")


@wrap_jit(static_argnames=["n1", "n2"])
def fft_interp2d(
    f: Num[ArrayLike, "nx ny ..."],
    n1: int,
    n2: int,
    sx: Optional[Num[ArrayLike, " s"]] = None,
    sy: Optional[Num[ArrayLike, " s"]] = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Inexact[Array, "n1 n2 ... s"]:
    """Interpolation of a real-valued 2D periodic function via FFT.

    Parameters
    ----------
    f : ndarray, shape(nx, ny, ...)
        Source data. Assumed to cover 1 full period, excluding the endpoint.
    n1, n2 : int
        Number of desired interpolation points in x and y directions
    sx, sy : ndarray or None
        Shift in x and y to evaluate at. If original data is f(x,y), interpolates to
        f(x + sx, y + sy). Both must be provided or None
    dx, dy : float
        Spacing of source points in x and y

    Returns
    -------
    fi : ndarray, shape(n1, n2, ..., len(sx))
        Interpolated (and possibly shifted) data points
    """
    f = asarray_inexact(f)
    c = jnp.fft.rfft2(f, axes=(0, 1), norm="forward")
    return _fft_interp2d(c, *f.shape[:2], n1, n2, sx, sy, dx, dy)


def _fft_interp2d(c, nx, ny, n1, n2, sx, sy, dx, dy):
    if (sx is not None) and (sy is not None):
        tau = 2 * jnp.pi
        sx = asarray_inexact(sx)
        sy = asarray_inexact(sy)
        sx = jnp.exp(1j * jnp.fft.fftfreq(nx, dx / tau)[:, None] * sx)
        sy = jnp.exp(1j * jnp.fft.rfftfreq(ny, dy / tau)[:, None] * sy)
        c = (c[None].T * (sx[None] * sy[:, None])).T
        c = jnp.moveaxis(c, 0, -1)

    return jnp.fft.irfft2(_fft_pad(c, n1, nx), (n1, n2), axes=(0, 1), norm="forward")


def _fft_pad(c, n_out, n_in, axis=0):
    p = n_out - n_in
    p = (p // 2, p - p // 2)
    if n_in % 2 != 0:
        p = p[::-1]
    return jnp.fft.ifftshift(_pad_along_axis(jnp.fft.fftshift(c, axis), p, axis), axis)


def _pad_along_axis(array: jax.Array, pad: tuple = (0, 0), axis: int = 0):
    """Pad with zeros or truncate a given dimension."""
    index = [slice(None)] * array.ndim
    pad_width = [(0, 0)] * array.ndim
    start = stop = None

    if pad[0] < 0:
        start = -pad[0]
        pad = (0, pad[1])
    if pad[1] < 0:
        stop = pad[1]
        pad = (pad[0], 0)

    index[axis] = slice(start, stop)
    pad_width[axis] = pad
    return jnp.pad(array[tuple(index)], pad_width)
