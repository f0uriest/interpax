from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames="n")
def fft_interp1d(f: jax.Array, n: int, sx: jax.Array = None, dx: float = 1.0):
    """Interpolation of a 1d periodic function via FFT.

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
    c = jnp.fft.ifft(f, axis=0)
    nx = c.shape[0]
    if sx is not None:
        sx = jnp.exp(-1j * 2 * jnp.pi * jnp.fft.fftfreq(nx)[:, None] * sx / dx)
        c = (c[None].T * sx).T
        c = jnp.moveaxis(c, 0, -1)
    pad = ((n - nx) // 2, n - nx - (n - nx) // 2)
    if nx % 2 != 0:
        pad = pad[::-1]
    c = jnp.fft.ifftshift(_pad_along_axis(jnp.fft.fftshift(c, axes=0), pad, axis=0))
    return jnp.fft.fft(c, axis=0).real


@partial(jit, static_argnames=("n1", "n2"))
def fft_interp2d(
    f: jax.Array,
    n1: int,
    n2: int,
    sx: jax.Array = None,
    sy: jax.Array = None,
    dx: float = 1.0,
    dy: float = 1.0,
):
    """Interpolation of a 2d periodic function via FFT.

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
    c = jnp.fft.ifft2(f, axes=(0, 1))
    nx, ny = c.shape[:2]
    if (sx is not None) and (sy is not None):
        sx = jnp.exp(-1j * 2 * jnp.pi * jnp.fft.fftfreq(nx)[:, None] * sx / dx)
        sy = jnp.exp(-1j * 2 * jnp.pi * jnp.fft.fftfreq(ny)[:, None] * sy / dy)
        c = (c[None].T * sx[None, :, :] * sy[:, None, :]).T
        c = jnp.moveaxis(c, 0, -1)
    padx = ((n1 - nx) // 2, n1 - nx - (n1 - nx) // 2)
    pady = ((n2 - ny) // 2, n2 - ny - (n2 - ny) // 2)
    if nx % 2 != 0:
        padx = padx[::-1]
    if ny % 2 != 0:
        pady = pady[::-1]

    c = jnp.fft.ifftshift(
        _pad_along_axis(jnp.fft.fftshift(c, axes=0), padx, axis=0), axes=0
    )
    c = jnp.fft.ifftshift(
        _pad_along_axis(jnp.fft.fftshift(c, axes=1), pady, axis=1), axes=1
    )

    return jnp.fft.fft2(c, axes=(0, 1)).real


def _pad_along_axis(array: jax.Array, pad: tuple = (0, 0), axis: int = 0):
    """Pad with zeros or truncate a given dimension."""
    array = jnp.moveaxis(array, axis, 0)

    if pad[0] < 0:
        array = array[abs(pad[0]) :]
        pad = (0, pad[1])
    if pad[1] < 0:
        array = array[: -abs(pad[1])]
        pad = (pad[0], 0)

    npad = [(0, 0)] * array.ndim
    npad[0] = pad

    array = jnp.pad(array, pad_width=npad, mode="constant", constant_values=0)
    return jnp.moveaxis(array, 0, axis)
