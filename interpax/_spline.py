"""Functions for interpolating splines that are JAX differentiable."""

from collections import OrderedDict
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from .utils import errorif, isbool

CUBIC_METHODS = ("cubic", "cubic2", "cardinal", "catmull-rom")
OTHER_METHODS = ("nearest", "linear")
METHODS_1D = CUBIC_METHODS + OTHER_METHODS + ("monotonic", "monotonic-0")
METHODS_2D = CUBIC_METHODS + OTHER_METHODS
METHODS_3D = CUBIC_METHODS + OTHER_METHODS


class Interpolator1D(eqx.Module):
    """Convenience class for representing a 1D interpolated function.

    Parameters
    ----------
    x : ndarray, shape(Nx,)
        coordinates of known function values ("knots")
    f : ndarray, shape(Nx,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints

    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as a 2 element array or tuple to specify different conditions
        for xq<x[0] and x[-1]<xq
    period : float > 0, None
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. None denotes no periodicity

    Notes
    -----
    This class is registered as a PyTree in JAX (it is actually an equinox.Module)
    so should be compatible with standard JAX transformations (jit, grad, vmap, etc.)

    """

    x: jax.Array
    f: jax.Array
    derivs: dict
    method: str
    extrap: bool | float | tuple
    period: float | tuple
    axis: int

    def __init__(self, x, f, method="cubic", extrap=False, period=None, **kwargs):
        x, f = map(jnp.asarray, (x, f))
        axis = kwargs.get("axis", 0)
        fx = kwargs.pop("fx", None)

        errorif(
            (len(x) != f.shape[axis]) or (jnp.ndim(x) != 1),
            ValueError,
            "x and f must be arrays of equal length",
        )
        errorif(method not in METHODS_1D, ValueError, f"unknown method {method}")

        self.x = x
        self.f = f
        self.axis = axis
        self.method = method
        self.extrap = extrap
        self.period = period

        if fx is None:
            fx = _approx_df(x, f, method, axis, **kwargs)

        self.derivs = {"fx": fx}

    def __call__(self, xq, dx=0):
        """Evaluate the interpolated function or its derivatives.

        Parameters
        ----------
        xq : ndarray, shape(Nq,)
            Query points where interpolation is desired
        dx : int >= 0
            Derivative to take.

        Returns
        -------
        fq : ndarray, shape(Nq, ...)
            Interpolated values.
        """
        return interp1d(
            xq,
            self.x,
            self.f,
            self.method,
            dx,
            self.extrap,
            self.period,
            **self.derivs,
        )


class Interpolator2D(eqx.Module):
    """Convenience class for representing a 2D interpolated function.

    Parameters
    ----------
    x : ndarray, shape(Nx,)
        x coordinates of known function values ("knots")
    y : ndarray, shape(Ny,)
        y coordinates of known function values ("knots")
    f : ndarray, shape(Nx,Ny,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension

    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions
        [[xlow, xhigh],[ylow,yhigh]]
    period : float > 0, None, array-like, shape(2,)
        periodicity of the function in x, y directions. None denotes no periodicity,
        otherwise function is assumed to be periodic on the interval [0,period]. Use a
        single value for the same in both directions.

    Notes
    -----
    This class is registered as a PyTree in JAX (it is actually an equinox.Module)
    so should be compatible with standard JAX transformations (jit, grad, vmap, etc.)

    """

    x: jax.Array
    y: jax.Array
    f: jax.Array
    derivs: dict
    method: str
    extrap: bool | float | tuple
    period: float | tuple
    axis: int

    def __init__(self, x, y, f, method="cubic", extrap=False, period=None, **kwargs):
        x, y, f = map(jnp.asarray, (x, y, f))
        axis = kwargs.get("axis", 0)
        fx = kwargs.pop("fx", None)
        fy = kwargs.pop("fy", None)
        fxy = kwargs.pop("fxy", None)

        errorif(
            (len(x) != f.shape[0]) or (x.ndim != 1),
            ValueError,
            "x and f must be arrays of equal length",
        )
        errorif(
            (len(y) != f.shape[1]) or (y.ndim != 1),
            ValueError,
            "y and f must be arrays of equal length",
        )
        errorif(method not in METHODS_2D, ValueError, f"unknown method {method}")

        self.x = x
        self.y = y
        self.f = f
        self.axis = axis
        self.method = method
        self.extrap = extrap
        self.period = period

        if fx is None:
            fx = _approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = _approx_df(y, f, method, 1, **kwargs)
        if fxy is None:
            fxy = _approx_df(y, fx, method, 1, **kwargs)

        self.derivs = {"fx": fx, "fy": fy, "fxy": fxy}

    def __call__(self, xq, yq, dx=0, dy=0):
        """Evaluate the interpolated function or its derivatives.

        Parameters
        ----------
        xq, yq : ndarray, shape(Nq,)
            x, y query points where interpolation is desired
        dx, dy : int >= 0
            Derivative to take in x, y directions.

        Returns
        -------
        fq : ndarray, shape(Nq, ...)
            Interpolated values.
        """
        return interp2d(
            xq,
            yq,
            self.x,
            self.y,
            self.f,
            self.method,
            (dx, dy),
            self.extrap,
            self.period,
            **self.derivs,
        )


class Interpolator3D(eqx.Module):
    """Convenience class for representing a 3D interpolated function.

    Parameters
    ----------
    x : ndarray, shape(Nx,)
        x coordinates of known function values ("knots")
    y : ndarray, shape(Ny,)
        y coordinates of known function values ("knots")
    z : ndarray, shape(Nz,)
        z coordinates of known function values ("knots")
    f : ndarray, shape(Nx,Ny,Nz,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension

    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions
        [[xlow, xhigh],[ylow,yhigh]]
    period : float > 0, None, array-like, shape(2,)
        periodicity of the function in x, y directions. None denotes no periodicity,
        otherwise function is assumed to be periodic on the interval [0,period]. Use a
        single value for the same in both directions.

    Notes
    -----
    This class is registered as a PyTree in JAX (it is actually an equinox.Module)
    so should be compatible with standard JAX transformations (jit, grad, vmap, etc.)

    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    f: jax.Array
    derivs: dict
    method: str
    extrap: bool | float | tuple
    period: float | tuple
    axis: int

    def __init__(self, x, y, z, f, method="cubic", extrap=False, period=None, **kwargs):
        x, y, z, f = map(jnp.asarray, (x, y, z, f))
        axis = kwargs.get("axis", 0)

        errorif(
            (len(x) != f.shape[0]) or (x.ndim != 1),
            ValueError,
            "x and f must be arrays of equal length",
        )
        errorif(
            (len(y) != f.shape[1]) or (y.ndim != 1),
            ValueError,
            "y and f must be arrays of equal length",
        )
        errorif(
            (len(z) != f.shape[2]) or (z.ndim != 1),
            ValueError,
            "z and f must be arrays of equal length",
        )
        errorif(method not in METHODS_3D, ValueError, f"unknown method {method}")

        fx = kwargs.pop("fx", None)
        fy = kwargs.pop("fy", None)
        fz = kwargs.pop("fz", None)
        fxy = kwargs.pop("fxy", None)
        fxz = kwargs.pop("fxz", None)
        fyz = kwargs.pop("fyz", None)
        fxyz = kwargs.pop("fxyz", None)

        self.x = x
        self.y = y
        self.z = z
        self.f = f
        self.axis = axis
        self.method = method
        self.extrap = extrap
        self.period = period

        if fx is None:
            fx = _approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = _approx_df(y, f, method, 1, **kwargs)
        if fz is None:
            fz = _approx_df(z, f, method, 2, **kwargs)
        if fxy is None:
            fxy = _approx_df(y, fx, method, 1, **kwargs)
        if fxz is None:
            fxz = _approx_df(z, fx, method, 2, **kwargs)
        if fyz is None:
            fyz = _approx_df(z, fy, method, 2, **kwargs)
        if fxyz is None:
            fxyz = _approx_df(z, fxy, method, 2, **kwargs)

        self.derivs = {
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "fxy": fxy,
            "fxz": fxz,
            "fyz": fyz,
            "fxyz": fxyz,
        }

    def __call__(self, xq, yq, zq, dx=0, dy=0, dz=0):
        """Evaluate the interpolated function or its derivatives.

        Parameters
        ----------
        xq, yq, zq : ndarray, shape(Nq,)
            x, y, z query points where interpolation is desired
        dx, dy, dz : int >= 0
            Derivative to take in x, y, z directions.

        Returns
        -------
        fq : ndarray, shape(Nq, ...)
            Interpolated values.
        """
        return interp3d(
            xq,
            yq,
            zq,
            self.x,
            self.y,
            self.z,
            self.f,
            self.method,
            (dx, dy, dz),
            self.extrap,
            self.period,
            **self.derivs,
        )


@partial(jit, static_argnames="method")
def interp1d(
    xq, x, f, method="cubic", derivative=0, extrap=False, period=None, **kwargs
):
    """Interpolate a 1d function.

    Parameters
    ----------
    xq : ndarray, shape(Nq,)
        query points where interpolation is desired
    x : ndarray, shape(Nx,)
        coordinates of known function values ("knots")
    f : ndarray, shape(Nx,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints

    derivative : int >= 0
        derivative order to calculate
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as a 2 element array or tuple to specify different conditions
        for xq<x[0] and x[-1]<xq
    period : float > 0, None
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. None denotes no periodicity

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points

    Notes
    -----
    For repeated interpolation given the same x, f data, recommend using Interpolator1D
    which caches the calculation of the derivatives and spline coefficients.

    """
    xq, x, f = map(jnp.asarray, (xq, x, f))
    axis = kwargs.get("axis", 0)
    fx = kwargs.pop("fx", None)

    errorif(
        (len(x) != f.shape[axis]) or (jnp.ndim(x) != 1),
        ValueError,
        "x and f must be arrays of equal length",
    )
    errorif(method not in METHODS_1D, ValueError, f"unknown method {method}")

    lowx, highx = _parse_extrap(extrap, 1)

    if period is not None:
        xq, x, f, fx = _make_periodic(xq, x, period, axis, f, fx)
        lowx = highx = True

    if method == "nearest":

        def derivative0():
            i = jnp.argmin(jnp.abs(xq[:, np.newaxis] - x[np.newaxis]), axis=1)
            return f[i]

        def derivative1():
            return jnp.zeros((xq.size, *f.shape[1:]))

        fq = jax.lax.switch(derivative, [derivative0, derivative1])

    elif method == "linear":

        def derivative0():
            i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
            df = jnp.take(f, i, axis) - jnp.take(f, i - 1, axis)
            dx = x[i] - x[i - 1]
            dxi = jnp.where(dx == 0, 0, 1 / dx)
            delta = xq - x[i - 1]
            fq = jnp.where(
                (dx == 0),
                jnp.take(f, i, axis),
                jnp.take(f, i - 1, axis) + delta * dxi * df,
            )
            return fq

        def derivative1():
            i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
            df = jnp.take(f, i, axis) - jnp.take(f, i - 1, axis)
            dx = x[i] - x[i - 1]
            dxi = jnp.where(dx == 0, 0, 1 / dx)
            return df * dxi

        def derivative2():
            return jnp.zeros((xq.size, *f.shape[1:]))

        fq = jax.lax.switch(derivative, [derivative0, derivative1, derivative2])

    elif method in (CUBIC_METHODS + ("monotonic", "monotonic-0")):

        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        if fx is None:
            fx = _approx_df(x, f, method, axis, **kwargs)

        dx = x[i] - x[i - 1]
        delta = xq - x[i - 1]
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        t = delta * dxi

        f0 = jnp.take(f, i - 1, axis)
        f1 = jnp.take(f, i, axis)
        fx0 = jnp.take(fx, i - 1, axis) * dx
        fx1 = jnp.take(fx, i, axis) * dx

        F = jnp.vstack([f0, f1, fx0, fx1])
        coef = jnp.matmul(A_CUBIC, F)
        ttx = _get_t_der(t, derivative, dxi)
        fq = jnp.einsum("ij,ji...->i...", ttx, coef)

    fq = _extrap(xq, fq, x, lowx, highx)
    return fq


@partial(jit, static_argnames="method")
def interp2d(  # noqa: C901 - FIXME: break this up into simpler pieces
    xq,
    yq,
    x,
    y,
    f,
    method="cubic",
    derivative=0,
    extrap=False,
    period=None,
    **kwargs,
):
    """Interpolate a 2d function.

    Parameters
    ----------
    xq : ndarray, shape(Nq,)
        x query points where interpolation is desired
    yq : ndarray, shape(Nq,)
        y query points where interpolation is desired
    x : ndarray, shape(Nx,)
        x coordinates of known function values ("knots")
    y : ndarray, shape(Ny,)
        y coordinates of known function values ("knots")
    f : ndarray, shape(Nx,Ny,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension

    derivative : int >= 0 or array-like, shape(2,)
        derivative order to calculate in x, y. Use a single value for the same in both
        directions.
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions
        [[xlow, xhigh],[ylow,yhigh]]
    period : float > 0, None, array-like, shape(2,)
        periodicity of the function in x, y directions. None denotes no periodicity,
        otherwise function is assumed to be periodic on the interval [0,period]. Use a
        single value for the same in both directions.

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points

    Notes
    -----
    For repeated interpolation given the same x, y, f data, recommend using
    Interpolator2D which caches the calculation of the derivatives and spline
    coefficients.

    """
    xq, yq, x, y, f = map(jnp.asarray, (xq, yq, x, y, f))
    fx = kwargs.pop("fx", None)
    fy = kwargs.pop("fy", None)
    fxy = kwargs.pop("fxy", None)
    xq, yq = jnp.broadcast_arrays(xq, yq)

    errorif(
        (len(x) != f.shape[0]) or (x.ndim != 1),
        ValueError,
        "x and f must be arrays of equal length",
    )
    errorif(
        (len(y) != f.shape[1]) or (y.ndim != 1),
        ValueError,
        "y and f must be arrays of equal length",
    )
    errorif(method not in METHODS_2D, ValueError, f"unknown method {method}")

    periodx, periody = _parse_ndarg(period, 2)
    derivative_x, derivative_y = _parse_ndarg(derivative, 2)
    lowx, highx, lowy, highy = _parse_extrap(extrap, 2)

    if periodx is not None:
        xq, x, f, fx, fy, fxy = _make_periodic(xq, x, periodx, 0, f, fx, fy, fxy)
        lowx = highx = True
    if periody is not None:
        yq, y, f, fx, fy, fxy = _make_periodic(yq, y, periody, 1, f, fx, fy, fxy)
        lowy = highy = True

    if method == "nearest":

        def derivative0():
            i = jnp.argmin(jnp.abs(xq[:, np.newaxis] - x[np.newaxis]), axis=1)
            j = jnp.argmin(jnp.abs(yq[:, np.newaxis] - y[np.newaxis]), axis=1)
            return f[i, j]

        def derivative1():
            return jnp.zeros((xq.size, *f.shape[2:]))

        fq = jax.lax.cond(
            (derivative_x == 0) & (derivative_y == 0), derivative0, derivative1
        )

    elif method == "linear":

        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)

        f00 = f[i - 1, j - 1]
        f01 = f[i - 1, j]
        f10 = f[i, j - 1]
        f11 = f[i, j]
        x0 = x[i - 1]
        x1 = x[i]
        y0 = y[j - 1]
        y1 = y[j]
        dx = x1 - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        dy = y1 - y0
        dyi = jnp.where(dy == 0, 0, 1 / dy)

        dx0 = lambda: jnp.array([x1 - xq, xq - x0])
        dx1 = lambda: jnp.array([-jnp.ones_like(xq), jnp.ones_like(xq)])
        dx2 = lambda: jnp.zeros((2, xq.size))
        dy0 = lambda: jnp.array([y1 - yq, yq - y0])
        dy1 = lambda: jnp.array([-jnp.ones_like(yq), jnp.ones_like(yq)])
        dy2 = lambda: jnp.zeros((2, yq.size))

        tx = jax.lax.switch(derivative_x, [dx0, dx1, dx2])
        ty = jax.lax.switch(derivative_y, [dy0, dy1, dy2])
        F = jnp.array([[f00, f01], [f10, f11]])
        fq = dxi * dyi * jnp.einsum("ik,ijk,jk->k", tx, F, ty)

    elif method in CUBIC_METHODS:

        if fx is None:
            fx = _approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = _approx_df(y, f, method, 1, **kwargs)
        if fxy is None:
            fxy = _approx_df(y, fx, method, 1, **kwargs)

        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)

        dx = x[i] - x[i - 1]
        deltax = xq - x[i - 1]
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        tx = deltax * dxi
        dy = y[j] - y[j - 1]
        deltay = yq - y[j - 1]
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        ty = deltay * dyi

        fs = OrderedDict()
        fs["f"] = f
        fs["fx"] = fx
        fs["fy"] = fy
        fs["fxy"] = fxy
        fsq = OrderedDict()
        for ff in fs.keys():
            for jj in [0, 1]:
                for ii in [0, 1]:
                    fsq[ff + str(ii) + str(jj)] = fs[ff][i - 1 + ii, j - 1 + jj]
                    if "x" in ff:
                        fsq[ff + str(ii) + str(jj)] *= dx
                    if "y" in ff:
                        fsq[ff + str(ii) + str(jj)] *= dy

        F = jnp.vstack([foo for foo in fsq.values()])
        coef = jnp.matmul(A_BICUBIC, F)
        coef = jnp.moveaxis(coef.reshape((4, 4, -1), order="F"), -1, 0)
        ttx = _get_t_der(tx, derivative_x, dxi)
        tty = _get_t_der(ty, derivative_y, dyi)
        fq = jnp.einsum("ij,ijk...,ik->i...", ttx, coef, tty)

    fq = _extrap(xq, fq, x, lowx, highx)
    fq = _extrap(yq, fq, y, lowy, highy)

    return fq


@partial(jit, static_argnames="method")
def interp3d(  # noqa: C901 - FIXME: break this up into simpler pieces
    xq,
    yq,
    zq,
    x,
    y,
    z,
    f,
    method="cubic",
    derivative=0,
    extrap=False,
    period=None,
    **kwargs,
):
    """Interpolate a 3d function.

    Parameters
    ----------
    xq : ndarray, shape(Nq,)
        x query points where interpolation is desired
    yq : ndarray, shape(Nq,)
        y query points where interpolation is desired
    zq : ndarray, shape(Nq,)
        z query points where interpolation is desired
    x : ndarray, shape(Nx,)
        x coordinates of known function values ("knots")
    y : ndarray, shape(Ny,)
        y coordinates of known function values ("knots")
    z : ndarray, shape(Nz,)
        z coordinates of known function values ("knots")
    f : ndarray, shape(Nx,Ny,Nz,...)
        function values to interpolate
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension

    derivative : int >= 0, array-like, shape(3,)
        derivative order to calculate in x,y,z directions. Use a single value for the
        same in all directions.
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions for
        [[xlow, xhigh],[ylow,yhigh],[zlow,zhigh]]
    period : float > 0, None, array-like, shape(3,)
        periodicity of the function in x, y, z directions. None denotes no periodicity,
        otherwise function is assumed to be periodic on the interval [0,period]. Use a
        single value for the same in all directions.

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points

    Notes
    -----
    For repeated interpolation given the same x, y, z, f data, recommend using
    Interpolator3D which caches the calculation of the derivatives and spline
    coefficients.

    """
    xq, yq, zq, x, y, z, f = map(jnp.asarray, (xq, yq, zq, x, y, z, f))
    errorif(
        (len(x) != f.shape[0]) or (x.ndim != 1),
        ValueError,
        "x and f must be arrays of equal length",
    )
    errorif(
        (len(y) != f.shape[1]) or (y.ndim != 1),
        ValueError,
        "y and f must be arrays of equal length",
    )
    errorif(
        (len(z) != f.shape[2]) or (z.ndim != 1),
        ValueError,
        "z and f must be arrays of equal length",
    )
    errorif(method not in METHODS_3D, ValueError, f"unknown method {method}")

    xq, yq, zq = jnp.broadcast_arrays(xq, yq, zq)

    fx = kwargs.pop("fx", None)
    fy = kwargs.pop("fy", None)
    fz = kwargs.pop("fz", None)
    fxy = kwargs.pop("fxy", None)
    fxz = kwargs.pop("fxz", None)
    fyz = kwargs.pop("fyz", None)
    fxyz = kwargs.pop("fxyz", None)

    periodx, periody, periodz = _parse_ndarg(period, 3)
    derivative_x, derivative_y, derivative_z = _parse_ndarg(derivative, 3)
    lowx, highx, lowy, highy, lowz, highz = _parse_extrap(extrap, 3)

    if periodx is not None:
        xq, x, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            xq, x, periodx, 0, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowx = highx = True
    if periody is not None:
        yq, y, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            yq, y, periody, 1, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowy = highy = True
    if periodz is not None:
        zq, z, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            zq, z, periodz, 2, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowz = highz = True

    if method == "nearest":

        def derivative0():
            i = jnp.argmin(jnp.abs(xq[:, np.newaxis] - x[np.newaxis]), axis=1)
            j = jnp.argmin(jnp.abs(yq[:, np.newaxis] - y[np.newaxis]), axis=1)
            k = jnp.argmin(jnp.abs(zq[:, np.newaxis] - z[np.newaxis]), axis=1)
            return f[i, j, k]

        def derivative1():
            return jnp.zeros((xq.size, *f.shape[3:]))

        fq = jax.lax.cond(
            (derivative_x == 0) & (derivative_y == 0) & (derivative_z == 0),
            derivative0,
            derivative1,
        )

    elif method == "linear":

        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
        k = jnp.clip(jnp.searchsorted(z, zq, side="right"), 1, len(z) - 1)

        f000 = f[i - 1, j - 1, k - 1]
        f001 = f[i - 1, j - 1, k]
        f010 = f[i - 1, j, k - 1]
        f100 = f[i, j - 1, k - 1]
        f110 = f[i, j, k - 1]
        f011 = f[i - 1, j, k]
        f101 = f[i, j - 1, k]
        f111 = f[i, j, k]
        x0 = x[i - 1]
        x1 = x[i]
        y0 = y[j - 1]
        y1 = y[j]
        z0 = z[k - 1]
        z1 = z[k]
        dx = x1 - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        dy = y1 - y0
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        dz = z1 - z0
        dzi = jnp.where(dz == 0, 0, 1 / dz)

        dx0 = lambda: jnp.array([x1 - xq, xq - x0])
        dx1 = lambda: jnp.array([-jnp.ones_like(xq), jnp.ones_like(xq)])
        dx2 = lambda: jnp.zeros((2, xq.size))
        dy0 = lambda: jnp.array([y1 - yq, yq - y0])
        dy1 = lambda: jnp.array([-jnp.ones_like(yq), jnp.ones_like(yq)])
        dy2 = lambda: jnp.zeros((2, yq.size))
        dz0 = lambda: jnp.array([z1 - zq, zq - z0])
        dz1 = lambda: jnp.array([-jnp.ones_like(zq), jnp.ones_like(zq)])
        dz2 = lambda: jnp.zeros((2, zq.size))

        tx = jax.lax.switch(derivative_x, [dx0, dx1, dx2])
        ty = jax.lax.switch(derivative_y, [dy0, dy1, dy2])
        tz = jax.lax.switch(derivative_z, [dz0, dz1, dz2])

        F = jnp.array([[[f000, f010], [f100, f110]], [[f001, f011], [f101, f111]]])
        fq = dxi * dyi * dzi * jnp.einsum("il,ijkl,jl,kl->l", tx, F, ty, tz)

    elif method in CUBIC_METHODS:
        if fx is None:
            fx = _approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = _approx_df(y, f, method, 1, **kwargs)
        if fz is None:
            fz = _approx_df(z, f, method, 2, **kwargs)
        if fxy is None:
            fxy = _approx_df(y, fx, method, 1, **kwargs)
        if fxz is None:
            fxz = _approx_df(z, fx, method, 2, **kwargs)
        if fyz is None:
            fyz = _approx_df(z, fy, method, 2, **kwargs)
        if fxyz is None:
            fxyz = _approx_df(z, fxy, method, 2, **kwargs)

        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
        k = jnp.clip(jnp.searchsorted(z, zq, side="right"), 1, len(z) - 1)

        dx = x[i] - x[i - 1]
        deltax = xq - x[i - 1]
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        tx = deltax * dxi

        dy = y[j] - y[j - 1]
        deltay = yq - y[j - 1]
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        ty = deltay * dyi

        dz = z[k] - z[k - 1]
        deltaz = zq - z[k - 1]
        dzi = jnp.where(dz == 0, 0, 1 / dz)
        tz = deltaz * dzi

        fs = OrderedDict()
        fs["f"] = f
        fs["fx"] = fx
        fs["fy"] = fy
        fs["fz"] = fz
        fs["fxy"] = fxy
        fs["fxz"] = fxz
        fs["fyz"] = fyz
        fs["fxyz"] = fxyz
        fsq = OrderedDict()
        for ff in fs.keys():
            for kk in [0, 1]:
                for jj in [0, 1]:
                    for ii in [0, 1]:
                        fsq[ff + str(ii) + str(jj) + str(kk)] = fs[ff][
                            i - 1 + ii, j - 1 + jj, k - 1 + kk
                        ]
                        if "x" in ff:
                            fsq[ff + str(ii) + str(jj) + str(kk)] *= dx
                        if "y" in ff:
                            fsq[ff + str(ii) + str(jj) + str(kk)] *= dy
                        if "z" in ff:
                            fsq[ff + str(ii) + str(jj) + str(kk)] *= dz

        F = jnp.vstack([foo for foo in fsq.values()])
        coef = jnp.matmul(A_TRICUBIC, F)
        coef = jnp.moveaxis(coef.reshape((4, 4, 4, -1), order="F"), -1, 0)
        ttx = _get_t_der(tx, derivative_x, dxi)
        tty = _get_t_der(ty, derivative_y, dyi)
        ttz = _get_t_der(tz, derivative_z, dzi)
        fq = jnp.einsum("lijk...,li,lj,lk->l...", coef, ttx, tty, ttz)

    fq = _extrap(xq, fq, x, lowx, highx)
    fq = _extrap(yq, fq, y, lowy, highy)
    fq = _extrap(zq, fq, z, lowz, highz)

    return fq


@partial(jit, static_argnames=("axis"))
def _make_periodic(xq, x, period, axis, *arrs):
    """Make arrays periodic along a specified axis."""
    period = abs(period)
    xq = xq % period
    x = x % period
    i = jnp.argsort(x)
    x = x[i]
    x = jnp.concatenate([x[-1:] - period, x, x[:1] + period])
    arrs = list(arrs)
    for k in range(len(arrs)):
        if arrs[k] is not None:
            arrs[k] = jnp.take(arrs[k], i, axis, mode="wrap")
            arrs[k] = jnp.concatenate(
                [
                    jnp.take(arrs[k], jnp.array([-1]), axis),
                    arrs[k],
                    jnp.take(arrs[k], jnp.array([0]), axis),
                ],
                axis=axis,
            )
    return (xq, x, *arrs)


@jit
def _get_t_der(t, derivative, dxi):
    """Get arrays of [1,t,t^2,t^3] for cubic interpolation."""
    t0 = jnp.zeros_like(t)
    t1 = jnp.ones_like(t)
    dxi = dxi[:, None]
    # derivatives of monomials
    d0 = lambda: jnp.array([t1, t, t**2, t**3]).T
    d1 = lambda: jnp.array([t0, t1, 2 * t, 3 * t**2]).T * dxi
    d2 = lambda: jnp.array([t0, t0, 2 * t1, 6 * t]).T * dxi**2
    d3 = lambda: jnp.array([t0, t0, t0, 6 * t1]).T * dxi**3
    d4 = lambda: jnp.array([t0, t0, t0, t0]).T

    return jax.lax.switch(derivative, [d0, d1, d2, d3, d4])


def _parse_ndarg(arg, n):
    try:
        k = len(arg)
    except TypeError:
        arg = tuple(arg for _ in range(n))
        k = n
    assert k == n, "got too many args"
    return arg


def _parse_extrap(extrap, n):
    if isbool(extrap):  # same for lower,upper in all dimensions
        return tuple(extrap for _ in range(2 * n))
    elif len(extrap) == 2 and jnp.isscalar(extrap[0]):  # same l,h for all dimensions
        return tuple(e for _ in range(n) for e in extrap)
    elif len(extrap) == n and all(len(extrap[i]) == 2 for i in range(n)):
        return tuple(eij for ei in extrap for eij in ei)
    else:
        raise ValueError(
            "extrap should either be a scalar, 2 element sequence (lo, hi), "
            + "or a sequence with 2 elements for each dimension"
        )


@jit
def _extrap(xq, fq, x, lo, hi):
    """Clamp or extrapolate values outside bounds."""

    def loclip(fq, lo):
        # lo is either False (no extrapolation) or a fixed value to fill in
        if isbool(lo):
            lo = jnp.nan
        return jnp.where(xq < x[0], lo, fq)

    def hiclip(fq, hi):
        # hi is either False (no extrapolation) or a fixed value to fill in
        if isbool(hi):
            hi = jnp.nan
        return jnp.where(xq > x[-1], hi, fq)

    def noclip(fq, *_):
        return fq

    fq = jax.lax.cond(
        isbool(lo) & lo,
        noclip,
        loclip,
        fq,
        lo,
    )
    fq = jax.lax.cond(
        isbool(hi) & hi,
        noclip,
        hiclip,
        fq,
        hi,
    )

    return fq


@partial(jit, static_argnames=("method", "axis"))
def _approx_df(x, f, method, axis, **kwargs):
    """Approximates derivatives for cubic spline interpolation."""
    if method == "cubic":
        dx = jnp.diff(x)
        df = jnp.diff(f, axis=axis)
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        if df.ndim > dxi.ndim:
            dxi = jnp.expand_dims(dxi, tuple(range(1, df.ndim)))
            dxi = jnp.moveaxis(dxi, 0, axis)
        df = dxi * df
        fx = jnp.concatenate(
            [
                jnp.take(df, jnp.array([0]), axis, mode="wrap"),
                1
                / 2
                * (
                    jnp.take(df, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                    + jnp.take(df, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                ),
                jnp.take(df, jnp.array([-1]), axis, mode="wrap"),
            ],
            axis=axis,
        )
        return fx

    elif method == "cubic2":
        dx = jnp.diff(x)
        df = jnp.diff(f, axis=axis)
        if df.ndim > dx.ndim:
            dx = jnp.expand_dims(dx, tuple(range(1, df.ndim)))
            dx = jnp.moveaxis(dx, 0, axis)
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        df = dxi * df

        A = jnp.diag(
            jnp.concatenate(
                (
                    np.array([1.0]),
                    2 * (dx.flatten()[:-1] + dx.flatten()[1:]),
                    np.array([1.0]),
                )
            )
        )
        upper_diag1 = jnp.diag(
            jnp.concatenate((np.array([1.0]), dx.flatten()[:-1])), k=1
        )
        lower_diag1 = jnp.diag(
            jnp.concatenate((dx.flatten()[1:], np.array([1.0]))), k=-1
        )
        A += upper_diag1 + lower_diag1
        b = jnp.concatenate(
            [
                2 * jnp.take(df, jnp.array([0]), axis, mode="wrap"),
                3
                * (
                    jnp.take(dx, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                    * jnp.take(df, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                    + jnp.take(dx, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                    * jnp.take(df, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                ),
                2 * jnp.take(df, jnp.array([-1]), axis, mode="wrap"),
            ],
            axis=axis,
        )
        b = jnp.moveaxis(b, axis, 0).reshape((b.shape[axis], -1))
        fx = jnp.linalg.solve(A, b)
        fx = jnp.moveaxis(fx.reshape(f.shape), 0, axis)
        return fx

    elif method in ["cardinal", "catmull-rom"]:
        dx = x[2:] - x[:-2]
        df = jnp.take(f, jnp.arange(2, f.shape[axis]), axis, mode="wrap") - jnp.take(
            f, jnp.arange(0, f.shape[axis] - 2), axis, mode="wrap"
        )
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        if df.ndim > dxi.ndim:
            dxi = jnp.expand_dims(dxi, tuple(range(1, df.ndim)))
            dxi = jnp.moveaxis(dxi, 0, axis)
        df = dxi * df
        fx0 = jnp.take(f, jnp.array([1]), axis, mode="wrap") - jnp.take(
            f, jnp.array([0]), axis, mode="wrap"
        )
        fx0 *= jnp.where(x[0] == x[1], 0, 1 / (x[1] - x[0]))
        fx1 = jnp.take(f, jnp.array([-1]), axis, mode="wrap") - jnp.take(
            f, jnp.array([-2]), axis, mode="wrap"
        )
        fx1 *= jnp.where(x[-1] == x[-2], 0, 1 / (x[-1] - x[-2]))

        if method == "cardinal":
            c = kwargs.get("c", 0)
        else:
            c = 0
        fx = (1 - c) * jnp.concatenate([fx0, df, fx1], axis=axis)
        return fx

    elif method in ["monotonic", "monotonic-0"]:
        f = jnp.moveaxis(f, axis, 0)
        fshp = f.shape
        if f.ndim == 1:
            # So that _edge_case doesn't end up assigning to scalars
            x = x[:, None]
            f = f[:, None]
        hk = x[1:] - x[:-1]
        df = jnp.diff(f, axis=axis)
        hki = jnp.where(hk == 0, 0, 1 / hk)
        if df.ndim > hki.ndim:
            hki = jnp.expand_dims(hki, tuple(range(1, df.ndim)))
            hki = jnp.moveaxis(hki, 0, axis)

        mk = hki * df

        smk = jnp.sign(mk)
        condition = (smk[1:, :] != smk[:-1, :]) | (mk[1:, :] == 0) | (mk[:-1, :] == 0)

        w1 = 2 * hk[1:] + hk[:-1]
        w2 = hk[1:] + 2 * hk[:-1]

        if df.ndim > w1.ndim:
            w1 = jnp.expand_dims(w1, tuple(range(1, df.ndim)))
            w1 = jnp.moveaxis(w1, 0, axis)
            w2 = jnp.expand_dims(w2, tuple(range(1, df.ndim)))
            w2 = jnp.moveaxis(w2, 0, axis)

        whmean = (w1 / mk[:-1, :] + w2 / mk[1:, :]) / (w1 + w2)

        dk = jnp.where(condition, 0, 1.0 / whmean)

        if method == "monotonic-0":
            d0 = jnp.zeros((1, dk.shape[1]))
            d1 = jnp.zeros((1, dk.shape[1]))

        else:
            # special case endpoints, as suggested in
            # Cleve Moler, Numerical Computing with MATLAB, Chap 3.6 (pchiptx.m)
            def _edge_case(h0, h1, m0, m1):
                # one-sided three-point estimate for the derivative
                d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

                # try to preserve shape
                mask = jnp.sign(d) != jnp.sign(m0)
                mask2 = (jnp.sign(m0) != jnp.sign(m1)) & (
                    jnp.abs(d) > 3.0 * jnp.abs(m0)
                )
                mmm = (~mask) & mask2

                d = jnp.where(mask, 0.0, d)
                d = jnp.where(mmm, 3.0 * m0, d)
                return d

            hk = 1 / hki
            d0 = _edge_case(hk[0, :], hk[1, :], mk[0, :], mk[1, :])[None]
            d1 = _edge_case(hk[-1, :], hk[-2, :], mk[-1, :], mk[-2, :])[None]

        dk = jnp.concatenate([d0, dk, d1])
        dk = dk.reshape(fshp)
        return dk.reshape(fshp)

    else:  # method passed in does not use df from this function, just return 0
        return jnp.zeros_like(f)


# fmt: off
A_TRICUBIC = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [-3, 3, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 2,-2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 9,-9,-9, 9, 0, 0, 0, 0, 6, 3,-6,-3, 0, 0, 0, 0, 6,-6, 3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [-6, 6, 6,-6, 0, 0, 0, 0,-3,-3, 3, 3, 0, 0, 0, 0,-4, 4,-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -2,-2,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [-6, 6, 6,-6, 0, 0, 0, 0,-4,-2, 4, 2, 0, 0, 0, 0,-3, 3,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -2,-1,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 4,-4,-4, 4, 0, 0, 0, 0, 2, 2,-2,-2, 0, 0, 0, 0, 2,-2, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,-9,-9, 9, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 6, 3,-6,-3, 0, 0, 0, 0, 6,-6, 3,-3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 6,-6, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0,-3,-3, 3, 3, 0, 0, 0, 0,-4, 4,-2, 2, 0, 0, 0, 0,-2,-2,-1,-1, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 6,-6, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0,-4,-2, 4, 2, 0, 0, 0, 0,-3, 3,-3, 3, 0, 0, 0, 0,-2,-1,-2,-1, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-4,-4, 4, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 2, 2,-2,-2, 0, 0, 0, 0, 2,-2, 2,-2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], # noqa: E501
    [-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 9,-9, 0, 0,-9, 9, 0, 0, 6, 3, 0, 0,-6,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,-6, 0, 0, 3,-3, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [-6, 6, 0, 0, 6,-6, 0, 0,-3,-3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 4, 0, 0,-2, 2, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0,-2,-2, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,-9, 0, 0,-9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     6, 3, 0, 0,-6,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,-6, 0, 0, 3,-3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 0, 0, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -3,-3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 4, 0, 0,-2, 2, 0, 0,-2,-2, 0, 0,-1,-1, 0, 0], # noqa: E501
    [ 9, 0,-9, 0,-9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0,-6, 0,-3, 0, 6, 0,-6, 0, 3, 0,-3, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 9, 0,-9, 0,-9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     6, 0, 3, 0,-6, 0,-3, 0, 6, 0,-6, 0, 3, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0], # noqa: E501
    [-27,27,27,-27,27,-27,-27,27,-18,-9,18, 9,18, 9,-18,-9,-18,18,-9, 9,18,-18, 9,-9,-18,18,18,-18,-9, 9, 9, # noqa: E501
     -9,-12,-6,-6,-3,12, 6, 6, 3,-12,-6,12, 6,-6,-3, 6, 3,-12,12,-6, 6,-6, 6,-3, 3,-8,-4,-4,-2,-4,-2,-2,-1], # noqa: E501
    [18,-18,-18,18,-18,18,18,-18, 9, 9,-9,-9,-9,-9, 9, 9,12,-12, 6,-6,-12,12,-6, 6,12,-12,-12,12, 6,-6,-6, # noqa: E501
     6, 6, 6, 3, 3,-6,-6,-3,-3, 6, 6,-6,-6, 3, 3,-3,-3, 8,-8, 4,-4, 4,-4, 2,-2, 4, 4, 2, 2, 2, 2, 1, 1], # noqa: E501
    [-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 3, 0, 3, 0,-4, 0, 4, 0,-2, 0, 2, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-2, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0,-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -3, 0,-3, 0, 3, 0, 3, 0,-4, 0, 4, 0,-2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-2, 0,-1, 0,-1, 0], # noqa: E501
    [18,-18,-18,18,-18,18,18,-18,12, 6,-12,-6,-12,-6,12, 6, 9,-9, 9,-9,-9, 9,-9, 9,12,-12,-12,12, 6,-6,-6, # noqa: E501
     6, 6, 3, 6, 3,-6,-3,-6,-3, 8, 4,-8,-4, 4, 2,-4,-2, 6,-6, 6,-6, 3,-3, 3,-3, 4, 2, 4, 2, 2, 1, 2, 1], # noqa: E501
    [-12,12,12,-12,12,-12,-12,12,-6,-6, 6, 6, 6, 6,-6,-6,-6, 6,-6, 6, 6,-6, 6,-6,-8, 8, 8,-8,-4, 4, 4,-4, # noqa: E501
     -3,-3,-3,-3, 3, 3, 3, 3,-4,-4, 4, 4,-2,-2, 2, 2,-4, 4,-4, 4,-2, 2,-2, 2,-2,-2,-2,-2,-1,-1,-1,-1], # noqa: E501
    [ 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [-6, 6, 0, 0, 6,-6, 0, 0,-4,-2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-3, 3, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 4,-4, 0, 0,-4, 4, 0, 0, 2, 2, 0, 0,-2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 2,-2, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 0, 0, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -4,-2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0,-2,-1, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-4, 0, 0,-4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     2, 2, 0, 0,-2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], # noqa: E501
    [-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0,-2, 0, 4, 0, 2, 0,-3, 0, 3, 0,-3, 0, 3, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0,-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     -4, 0,-2, 0, 4, 0, 2, 0,-3, 0, 3, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0,-2, 0,-1, 0], # noqa: E501
    [18,-18,-18,18,-18,18,18,-18,12, 6,-12,-6,-12,-6,12, 6,12,-12, 6,-6,-12,12,-6, 6, 9,-9,-9, 9, 9,-9,-9, # noqa: E501
     9, 8, 4, 4, 2,-8,-4,-4,-2, 6, 3,-6,-3, 6, 3,-6,-3, 6,-6, 3,-3, 6,-6, 3,-3, 4, 2, 2, 1, 4, 2, 2, 1], # noqa: E501
    [-12,12,12,-12,12,-12,-12,12,-6,-6, 6, 6, 6, 6,-6,-6,-8, 8,-4, 4, 8,-8, 4,-4,-6, 6, 6,-6,-6, 6, 6,-6, # noqa: E501
     -4,-4,-2,-2, 4, 4, 2, 2,-3,-3, 3, 3,-3,-3, 3, 3,-4, 4,-2, 2,-4, 4,-2, 2,-2,-2,-1,-1,-2,-2,-1,-1], # noqa: E501
    [ 4, 0,-4, 0,-4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,-2, 0,-2, 0, 2, 0,-2, 0, 2, 0,-2, 0, # noqa: E501
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # noqa: E501
    [ 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,-4, 0,-4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # noqa: E501
     2, 0, 2, 0,-2, 0,-2, 0, 2, 0,-2, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], # noqa: E501
    [-12,12,12,-12,12,-12,-12,12,-8,-4, 8, 4, 8, 4,-8,-4,-6, 6,-6, 6, 6,-6, 6,-6,-6, 6, 6,-6,-6, 6, 6,-6, # noqa: E501
     -4,-2,-4,-2, 4, 2, 4, 2,-4,-2, 4, 2,-4,-2, 4, 2,-3, 3,-3, 3,-3, 3,-3, 3,-2,-1,-2,-1,-2,-1,-2,-1], # noqa: E501
    [ 8,-8,-8, 8,-8, 8, 8,-8, 4, 4,-4,-4,-4,-4, 4, 4, 4,-4, 4,-4,-4, 4,-4, 4, 4,-4,-4, 4, 4,-4,-4, 4, # noqa: E501
     2, 2, 2, 2,-2,-2,-2,-2, 2, 2,-2,-2, 2, 2,-2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 1, 1, 1, 1, 1, 1, 1, 1] # noqa: E501
])

A_BICUBIC = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0 ],
    [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0 ],
    [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1 ],
    [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1 ],
    [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0 ],
    [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1 ],
    [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]
])

A_CUBIC = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [-3, 3, -2, -1],
    [2, -2, 1, 1],
])
