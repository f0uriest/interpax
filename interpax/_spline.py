"""Functions for interpolating splines that are JAX differentiable."""

from collections import OrderedDict
from functools import partial
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from ._coefs import A_BICUBIC, A_CUBIC, A_TRICUBIC
from ._fd_derivs import approx_df
from .utils import asarray_inexact, errorif, isbool

CUBIC_METHODS = (
    "cubic",
    "cubic2",
    "cardinal",
    "catmull-rom",
    "akima",
    "monotonic",
    "monotonic-0",
)
OTHER_METHODS = ("nearest", "linear")
METHODS_1D = CUBIC_METHODS + OTHER_METHODS
METHODS_2D = CUBIC_METHODS + OTHER_METHODS
METHODS_3D = CUBIC_METHODS + OTHER_METHODS


class AbstractInterpolator(eqx.Module):
    """ABC convenience class for representing an interpolated function.

    Subclasses should implement the `__call__` method to evaluate the
    interpolated function.

    """

    f: eqx.AbstractVar[jax.Array]  # function values to interpolate
    derivs: eqx.AbstractVar[dict[str, jax.Array]]
    method: str = eqx.field(static=True)
    extrap: eqx.AbstractVar[Union[bool, float, tuple]]
    period: eqx.AbstractVar[Union[None, float, tuple]]
    axis: eqx.AbstractVar[int]


class Interpolator1D(AbstractInterpolator):
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
        - ``'akima'``: C1 cubic splines that appear smooth and natural

    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as a 2 element array or tuple to specify different conditions
        for xq<x[0] and x[-1]<xq
    period : float > 0, None
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. None denotes no periodicity

    """

    x: jax.Array
    f: jax.Array
    derivs: dict
    method: str = eqx.field(static=True)
    extrap: Union[bool, float, tuple]
    period: Union[None, float]
    axis: int

    def __init__(
        self,
        x: jax.Array,
        f: jax.Array,
        method: str = "cubic",
        extrap: Union[bool, float, tuple] = False,
        period: Union[None, float] = None,
        **kwargs,
    ):
        x, f = map(asarray_inexact, (x, f))
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
            fx = approx_df(x, f, method, axis, **kwargs)

        self.derivs = {"fx": fx}

    def __call__(self, xq: jax.Array, dx: int = 0):
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


class Interpolator2D(AbstractInterpolator):
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
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints
        - ``'akima'``: C1 cubic splines that appear smooth and natural

    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions
        [[xlow, xhigh],[ylow,yhigh]]
    period : float > 0, None, array-like, shape(2,)
        periodicity of the function in x, y directions. None denotes no periodicity,
        otherwise function is assumed to be periodic on the interval [0,period]. Use a
        single value for the same in both directions.

    """

    x: jax.Array
    y: jax.Array
    f: jax.Array
    derivs: dict
    method: str = eqx.field(static=True)
    extrap: Union[bool, float, tuple]
    period: Union[None, float, tuple]
    axis: int

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        f: jax.Array,
        method: str = "cubic",
        extrap: Union[bool, float, tuple] = False,
        period: Union[None, float, tuple] = None,
        **kwargs,
    ):
        x, y, f = map(asarray_inexact, (x, y, f))
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
            fx = approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = approx_df(y, f, method, 1, **kwargs)
        if fxy is None:
            fxy = approx_df(y, fx, method, 1, **kwargs)

        self.derivs = {"fx": fx, "fy": fy, "fxy": fxy}

    def __call__(self, xq: jax.Array, yq: jax.Array, dx: int = 0, dy: int = 0):
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


class Interpolator3D(AbstractInterpolator):
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
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints
        - ``'akima'``: C1 cubic splines that appear smooth and natural

    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions
        [[xlow, xhigh],[ylow,yhigh]]
    period : float > 0, None, array-like, shape(2,)
        periodicity of the function in x, y, z directions. None denotes no periodicity,
        otherwise function is assumed to be periodic on the interval [0,period]. Use a
        single value for the same in both directions.

    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    f: jax.Array
    derivs: dict
    method: str = eqx.field(static=True)
    extrap: Union[bool, float, tuple]
    period: Union[None, float, tuple]
    axis: int

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        z: jax.Array,
        f: jax.Array,
        method: str = "cubic",
        extrap: Union[bool, float, tuple] = False,
        period: Union[None, float, tuple] = None,
        **kwargs,
    ):
        x, y, z, f = map(asarray_inexact, (x, y, z, f))
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
            fx = approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = approx_df(y, f, method, 1, **kwargs)
        if fz is None:
            fz = approx_df(z, f, method, 2, **kwargs)
        if fxy is None:
            fxy = approx_df(y, fx, method, 1, **kwargs)
        if fxz is None:
            fxz = approx_df(z, fx, method, 2, **kwargs)
        if fyz is None:
            fyz = approx_df(z, fy, method, 2, **kwargs)
        if fxyz is None:
            fxyz = approx_df(z, fxy, method, 2, **kwargs)

        self.derivs = {
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "fxy": fxy,
            "fxz": fxz,
            "fyz": fyz,
            "fxyz": fxyz,
        }

    def __call__(
        self,
        xq: jax.Array,
        yq: jax.Array,
        zq: jax.Array,
        dx: int = 0,
        dy: int = 0,
        dz: int = 0,
    ):
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
    xq: jax.Array,
    x: jax.Array,
    f: jax.Array,
    method: str = "cubic",
    derivative: int = 0,
    extrap: Union[bool, float, tuple] = False,
    period: Union[None, float] = None,
    **kwargs,
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
        - ``'akima'``: C1 cubic splines that appear smooth and natural

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
    xq, x, f = map(asarray_inexact, (xq, x, f))
    axis = kwargs.get("axis", 0)
    fx = kwargs.pop("fx", None)
    outshape = xq.shape + f.shape[1:]

    # Promote scalar query points to 1D array.
    # Note this is done after the computation of outshape
    # to make jax.grad work in the scalar case.
    xq = jnp.atleast_1d(xq)

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
                jnp.take(f, i, axis).T,
                jnp.take(f, i - 1, axis).T + (delta * dxi * df.T),
            ).T
            return fq

        def derivative1():
            i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
            df = jnp.take(f, i, axis) - jnp.take(f, i - 1, axis)
            dx = x[i] - x[i - 1]
            dxi = jnp.where(dx == 0, 0, 1 / dx)
            return (df.T * dxi).T

        def derivative2():
            return jnp.zeros((xq.size, *f.shape[1:]))

        fq = jax.lax.switch(derivative, [derivative0, derivative1, derivative2])

    elif method in CUBIC_METHODS:

        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        if fx is None:
            fx = approx_df(x, f, method, axis, **kwargs)
        assert fx.shape == f.shape

        dx = x[i] - x[i - 1]
        delta = xq - x[i - 1]
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        t = delta * dxi

        f0 = jnp.take(f, i - 1, axis)
        f1 = jnp.take(f, i, axis)
        fx0 = (jnp.take(fx, i - 1, axis).T * dx).T
        fx1 = (jnp.take(fx, i, axis).T * dx).T

        F = jnp.stack([f0, f1, fx0, fx1], axis=0).T
        coef = jnp.vectorize(jnp.matmul, signature="(n,n),(n)->(n)")(A_CUBIC, F).T
        ttx = _get_t_der(t, derivative, dxi)
        fq = jnp.einsum("ji...,ij->i...", coef, ttx)

    fq = _extrap(xq, fq, x, lowx, highx)
    return fq.reshape(outshape)


@partial(jit, static_argnames="method")
def interp2d(  # noqa: C901 - FIXME: break this up into simpler pieces
    xq: jax.Array,
    yq: jax.Array,
    x: jax.Array,
    y: jax.Array,
    f: jax.Array,
    method: str = "cubic",
    derivative: int = 0,
    extrap: Union[bool, float, tuple] = False,
    period: Union[None, float, tuple] = None,
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
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints
        - ``'akima'``: C1 cubic splines that appear smooth and natural

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
    xq, yq, x, y, f = map(asarray_inexact, (xq, yq, x, y, f))
    fx = kwargs.pop("fx", None)
    fy = kwargs.pop("fy", None)
    fxy = kwargs.pop("fxy", None)
    xq, yq = jnp.broadcast_arrays(xq, yq)
    outshape = xq.shape + f.shape[2:]

    # Promote scalar query points to 1D array.
    # Note this is done after the computation of outshape
    # to make jax.grad work in the scalar case.
    xq, yq = map(jnp.atleast_1d, (xq, yq))

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
            # because of the regular spaced grid we know that the nearest point
            # will be one of the 4 neighbors on the grid, so we first find those
            # and then take the nearest one among them.
            i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
            j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
            neighbors_x = jnp.array(
                [[x[i], x[i - 1], x[i], x[i - 1]], [y[j], y[j], y[j - 1], y[j - 1]]]
            )
            neighbors_f = jnp.array(
                [f[i, j].T, f[i - 1, j].T, f[i, j - 1].T, f[i - 1, j - 1].T]
            )
            xyq = jnp.array([xq, yq])
            dist = jnp.linalg.norm(neighbors_x - xyq[:, None, :], axis=0)
            idx = jnp.argmin(dist, axis=0)
            return jax.vmap(lambda a, b: jnp.take(a, b, axis=-1))(neighbors_f.T, idx)

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
        fq = (dxi * dyi * jnp.einsum("ijk...,ik,jk->k...", F, tx, ty).T).T

    elif method in CUBIC_METHODS:
        if fx is None:
            fx = approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = approx_df(y, f, method, 1, **kwargs)
        if fxy is None:
            fxy = approx_df(y, fx, method, 1, **kwargs)
        assert fx.shape == fy.shape == fxy.shape == f.shape

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
                    s = ff + str(ii) + str(jj)
                    fsq[s] = fs[ff][i - 1 + ii, j - 1 + jj]
                    if "x" in ff:
                        fsq[s] = (dx * fsq[s].T).T
                    if "y" in ff:
                        fsq[s] = (dy * fsq[s].T).T

        F = jnp.stack([foo for foo in fsq.values()], axis=0).T
        coef = jnp.vectorize(jnp.matmul, signature="(n,n),(n)->(n)")(A_BICUBIC, F).T
        coef = jnp.moveaxis(coef.reshape((4, 4, *coef.shape[1:]), order="F"), 2, 0)
        ttx = _get_t_der(tx, derivative_x, dxi)
        tty = _get_t_der(ty, derivative_y, dyi)
        fq = jnp.einsum("ijk...,ij,ik->i...", coef, ttx, tty)

    fq = _extrap(xq, fq, x, lowx, highx)
    fq = _extrap(yq, fq, y, lowy, highy)

    return fq.reshape(outshape)


@partial(jit, static_argnames="method")
def interp3d(  # noqa: C901 - FIXME: break this up into simpler pieces
    xq: jax.Array,
    yq: jax.Array,
    zq: jax.Array,
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    f: jax.Array,
    method: str = "cubic",
    derivative: int = 0,
    extrap: Union[bool, float, tuple] = False,
    period: Union[None, float, tuple] = None,
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
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints
        - ``'akima'``: C1 cubic splines that appear smooth and natural

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
    xq, yq, zq, x, y, z, f = map(asarray_inexact, (xq, yq, zq, x, y, z, f))
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
    outshape = xq.shape + f.shape[3:]

    # Promote scalar query points to 1D array.
    # Note this is done after the computation of outshape
    # to make jax.grad work in the scalar case.
    xq, yq, zq = map(jnp.atleast_1d, (xq, yq, zq))

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
            # because of the regular spaced grid we know that the nearest point
            # will be one of the 8 neighbors on the grid, so we first find those
            # and then take the nearest one among them.
            i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
            j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
            k = jnp.clip(jnp.searchsorted(z, zq, side="right"), 1, len(z) - 1)
            neighbors_x = jnp.array(
                [
                    [x[i], x[i - 1], x[i], x[i - 1], x[i], x[i - 1], x[i], x[i - 1]],
                    [y[j], y[j], y[j - 1], y[j - 1], y[j], y[j], y[j - 1], y[j - 1]],
                    [z[k], z[k], z[k], z[k], z[k - 1], z[k - 1], z[k - 1], z[k - 1]],
                ]
            )
            neighbors_f = jnp.array(
                [
                    f[i, j, k].T,
                    f[i - 1, j, k].T,
                    f[i, j - 1, k].T,
                    f[i - 1, j - 1, k].T,
                    f[i, j, k - 1].T,
                    f[i - 1, j, k - 1].T,
                    f[i, j - 1, k - 1].T,
                    f[i - 1, j - 1, k - 1].T,
                ]
            )
            xyzq = jnp.array([xq, yq, zq])
            dist = jnp.linalg.norm(neighbors_x - xyzq[:, None, :], axis=0)
            idx = jnp.argmin(dist, axis=0)
            return jax.vmap(lambda a, b: jnp.take(a, b, axis=-1))(neighbors_f.T, idx)

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

        F = jnp.array([[[f000, f001], [f010, f011]], [[f100, f101], [f110, f111]]])
        fq = (dxi * dyi * dzi * jnp.einsum("lijk...,lk,ik,jk->k...", F, tx, ty, tz).T).T

    elif method in CUBIC_METHODS:
        if fx is None:
            fx = approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = approx_df(y, f, method, 1, **kwargs)
        if fz is None:
            fz = approx_df(z, f, method, 2, **kwargs)
        if fxy is None:
            fxy = approx_df(y, fx, method, 1, **kwargs)
        if fxz is None:
            fxz = approx_df(z, fx, method, 2, **kwargs)
        if fyz is None:
            fyz = approx_df(z, fy, method, 2, **kwargs)
        if fxyz is None:
            fxyz = approx_df(z, fxy, method, 2, **kwargs)
        assert (
            fx.shape
            == fy.shape
            == fz.shape
            == fxy.shape
            == fxz.shape
            == fyz.shape
            == fxyz.shape
            == f.shape
        )
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
                        s = ff + str(ii) + str(jj) + str(kk)
                        fsq[s] = fs[ff][i - 1 + ii, j - 1 + jj, k - 1 + kk]
                        if "x" in ff:
                            fsq[s] = (dx * fsq[s].T).T
                        if "y" in ff:
                            fsq[s] = (dy * fsq[s].T).T
                        if "z" in ff:
                            fsq[s] = (dz * fsq[s].T).T

        F = jnp.stack([foo for foo in fsq.values()], axis=0).T
        coef = jnp.vectorize(jnp.matmul, signature="(n,n),(n)->(n)")(A_TRICUBIC, F).T
        coef = jnp.moveaxis(coef.reshape((4, 4, 4, *coef.shape[1:]), order="F"), 3, 0)
        ttx = _get_t_der(tx, derivative_x, dxi)
        tty = _get_t_der(ty, derivative_y, dyi)
        ttz = _get_t_der(tz, derivative_z, dzi)
        fq = jnp.einsum("lijk...,li,lj,lk->l...", coef, ttx, tty, ttz)

    fq = _extrap(xq, fq, x, lowx, highx)
    fq = _extrap(yq, fq, y, lowy, highy)
    fq = _extrap(zq, fq, z, lowz, highz)

    return fq.reshape(outshape)


@partial(jit, static_argnames=("axis"))
def _make_periodic(xq: jax.Array, x: jax.Array, period: float, axis: int, *arrs):
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
def _get_t_der(t: jax.Array, derivative: int, dxi: jax.Array):
    """Get arrays of [1,t,t^2,t^3] for cubic interpolation."""
    t0 = jnp.zeros_like(t)
    t1 = jnp.ones_like(t)
    dxi = jnp.atleast_1d(dxi)[:, None]
    # derivatives of monomials
    d0 = lambda: jnp.array([t1, t, t**2, t**3]).T * dxi**0
    d1 = lambda: jnp.array([t0, t1, 2 * t, 3 * t**2]).T * dxi
    d2 = lambda: jnp.array([t0, t0, 2 * t1, 6 * t]).T * dxi**2
    d3 = lambda: jnp.array([t0, t0, t0, 6 * t1]).T * dxi**3
    d4 = lambda: jnp.array([t0, t0, t0, t0]).T * (dxi * 0)

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
    elif jnp.isscalar(extrap):
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
def _extrap(
    xq: jax.Array,
    fq: jax.Array,
    x: jax.Array,
    lo: Union[bool, float],
    hi: Union[bool, float],
):
    """Clamp or extrapolate values outside bounds."""

    def loclip(fq, lo):
        # lo is either False (no extrapolation) or a fixed value to fill in
        if isbool(lo):
            lo = jnp.nan
        return jnp.where(xq < x[0], lo, fq.T).T

    def hiclip(fq, hi):
        # hi is either False (no extrapolation) or a fixed value to fill in
        if isbool(hi):
            hi = jnp.nan
        return jnp.where(xq > x[-1], hi, fq.T).T

    def noclip(fq, *_):
        return fq

    # if extrap = True, don't clip. If it's false or numeric, clip to that value
    # isbool(x) & bool(x) is testing if extrap is True but works for np/jnp bools
    fq = jax.lax.cond(
        isbool(lo) & jnp.asarray(lo).astype(bool),
        noclip,
        loclip,
        fq,
        lo,
    )
    fq = jax.lax.cond(
        isbool(hi) & jnp.asarray(hi).astype(bool),
        noclip,
        hiclip,
        fq,
        hi,
    )

    return fq
