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
        Shift in x to evaluate at. If original data is f(x), interpolates to f(x + sx).
    dx : float
        Spacing of source points.

    Returns
    -------
    fi : ndarray, shape(n, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    f = asarray_inexact(f)
    return ifft_interp1d(jnp.fft.rfft(f, axis=0, norm="forward"), f.shape[0], n, sx, dx)


def ifft_interp1d(
    c,
    nx: int,
    n: int,
    sx: Optional[Num[ArrayLike, " s"]] = None,
    dx: float = 1.0,
) -> Inexact[Array, "n ... s"]:
    """Interpolation of a 1D Hermitian Fourier series via FFT.

    Parameters
    ----------
    c : ndarray, shape(nx // 2 + 1, ...)
        Fourier coefficients ``jnp.fft.rfft(f,axis=0,norm="forward")``.
    nx : bool
        Number of sample points e.g. ``f.shape[0]``.
    n : int
        Number of desired interpolation points.
    sx : ndarray or None
        Shift in x to evaluate at. If original data is f(x), interpolates to f(x + sx).
    dx : float
        Spacing of source points.

    Returns
    -------
    fi : ndarray, shape(n, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    if sx is not None:
        tau = 2 * jnp.pi
        sx = asarray_inexact(sx)
        sx = jnp.exp(1j * jnp.fft.rfftfreq(nx, dx / tau)[:, None] * sx)
        c = (c[None].T * sx).T
        c = jnp.moveaxis(c, 0, -1)

    if n >= nx:
        return jnp.fft.irfft(c, n, axis=0, norm="forward")

    if n < c.shape[0]:
        c = c[:n]
    elif nx % 2 == 0:
        c = c.at[-1].divide(2)
    c = c.at[0].divide(2) * 2

    x = jnp.linspace(0, dx * nx, n, endpoint=False)
    x = jnp.exp(1j * (c.shape[0] // 2) * x).reshape(n, *((1,) * (c.ndim - 1)))

    c = _fft_pad(c, n, 0)
    return (jnp.fft.ifft(c, axis=0, norm="forward") * x).real


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
        Number of desired interpolation points in x and y directions.
    sx, sy : ndarray or None
        Shift in x and y to evaluate at. If original data is f(x,y), interpolates to
        f(x + sx, y + sy). Both must be provided or None.
    dx, dy : float
        Spacing of source points in x and y.

    Returns
    -------
    fi : ndarray, shape(n1, n2, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    nx, ny = f.shape[:2]

    # https://github.com/f0uriest/interpax/pull/117
    if n1 < nx:
        f = fft_interp1d(f, n1, sx, dx)
        f = fft_interp1d(f.swapaxes(0, 1), n2, sy, dy).swapaxes(0, 1)
        return f
    if n2 < ny:
        f = fft_interp1d(f.swapaxes(0, 1), n2, sy, dy).swapaxes(0, 1)
        f = fft_interp1d(f, n1, sx, dx)
        return f

    return ifft_interp2d(
        jnp.fft.rfft2(asarray_inexact(f), axes=(0, 1), norm="forward"),
        ny,
        n1,
        n2,
        sx,
        sy,
        dx,
        dy,
    )


def ifft_interp2d(
    c,
    ny: int,
    n1: int,
    n2: int,
    sx: Optional[Num[ArrayLike, " s"]] = None,
    sy: Optional[Num[ArrayLike, " s"]] = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Inexact[Array, "n1 n2 ... s"]:
    """Interpolation of 2D Hermitian Fourier series via FFT.

    Parameters
    ----------
    c : ndarray, shape(nx, ny // 2 + 1, ...)
        Fourier coefficients ``jnp.fft.rfft2(f,axis=(0,1),norm="forward")``.
    ny : bool
        Number of sample points in y coordinate, e.g. ``f.shape[1]``.
    n1, n2 : int
        Number of desired interpolation points in x and y directions.
    sx, sy : ndarray or None
        Shift in x and y to evaluate at. If original data is f(x,y), interpolates to
        f(x + sx, y + sy). Both must be provided or None.
    dx, dy : float
        Spacing of source points in x and y.

    Returns
    -------
    fi : ndarray, shape(n1, n2, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    if (sx is not None) and (sy is not None):
        tau = 2 * jnp.pi
        sx = asarray_inexact(sx)
        sy = asarray_inexact(sy)
        sx = jnp.exp(1j * jnp.fft.fftfreq(c.shape[0], dx / tau)[:, None] * sx)
        sy = jnp.exp(1j * jnp.fft.rfftfreq(ny, dy / tau)[:, None] * sy)
        c = (c[None].T * (sx[None] * sy[:, None])).T
        c = jnp.moveaxis(c, 0, -1)

    c = _fft_pad(jnp.fft.fftshift(c, 0), n1, 0)
    if n2 >= ny:
        return jnp.fft.irfft2(c, (n1, n2), axes=(0, 1), norm="forward")

    if n2 < c.shape[1]:
        c = c[:, :n2]
    elif ny % 2 == 0:
        c = c.at[:, -1].divide(2)
    c = c.at[:, 0].divide(2) * 2

    y = jnp.linspace(0, dy * ny, n2, endpoint=False)
    y = jnp.exp(1j * (c.shape[1] // 2) * y).reshape(1, n2, *((1,) * (c.ndim - 2)))

    c = jnp.fft.ifft(c, axis=0, norm="forward")
    c = _fft_pad(c, n2, 1)
    return (jnp.fft.ifft(c, axis=1, norm="forward") * y).real


def _fft_pad(c_shift, n_out, axis):
    n_in = c_shift.shape[axis]
    p = n_out - n_in
    p = (p // 2, p - p // 2)
    if n_in % 2 != 0:
        p = p[::-1]
    return jnp.fft.ifftshift(_pad_along_axis(c_shift, p, axis), axis)


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
