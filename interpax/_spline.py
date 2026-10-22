"""Functions for interpolating splines that are JAX differentiable."""

import functools
import itertools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jaxtyping import Array, ArrayLike, Float, Inexact, Num, Real

from ._coefs import A_BICUBIC, A_CUBIC, A_TRICUBIC
from ._fd_derivs import approx_df
from .utils import asarray_inexact, errorif, isbool, wrap_jit

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
METHODS = CUBIC_METHODS + OTHER_METHODS


class AbstractInterpolator(eqx.Module):
    """ABC convenience class for representing an interpolated function.

    Subclasses should implement the `__call__` method to evaluate the
    interpolated function.

    """

    f: eqx.AbstractVar[Inexact[Array, "..."]]  # function values to interpolate
    derivs: eqx.AbstractVar[dict[str, Inexact[Array, "..."]]]
    method: str = eqx.field(static=True)
    extrap: eqx.AbstractVar[bool | float | tuple]
    period: eqx.AbstractVar[None | float | tuple]
    axis: eqx.AbstractVar[int | tuple[int, ...]]


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
    even_spacing : bool, None
        Whether x is evenly spaced, which allows a faster lookup of the interval
        containing each query point. If None, this is checked once at construction,
        or on each call if constructed inside a jitted function. True skips the check,
        and gives incorrect results if x is not evenly spaced. False always uses a
        binary search. Defaults to None. If periodic, even spacing requires x to cover
        one full period without the endpoint, e.g.
        ``np.linspace(0, period, N, endpoint=False)``.
    fx : ndarray, optional
        first derivatives of f with respect to x at the knots. For cubic methods, if
        given these are used in place of the derivatives approximated by ``method``.
    axis : int, optional
        axis of f along which x varies. Default is 0. The output has the query shape
        in place of this axis, ie
        ``fq.shape == f.shape[:axis] + xq.shape + f.shape[axis + 1 :]``.

    """

    x: Float[Array, " Nx"]
    f: Inexact[Array, " Nx ..."]
    derivs: dict
    method: str = eqx.field(static=True)
    extrap: bool | float | tuple
    period: None | float
    axis: int | tuple[int, ...] = eqx.field(static=True)
    even_spacing: bool | None = eqx.field(static=True)

    def __init__(
        self,
        x: Real[ArrayLike, " Nx"],
        f: Num[ArrayLike, " Nx ..."],
        method: str = "cubic",
        extrap: bool | float | tuple = False,
        period: None | float = None,
        even_spacing: bool | None = None,
        **kwargs,
    ) -> None:
        x, f = map(asarray_inexact, (x, f))
        (axis,) = _parse_axis(kwargs.pop("axis", None), 1, f.ndim)
        fx = kwargs.pop("fx", None)

        errorif(
            (len(x) != f.shape[axis]) or (jnp.ndim(x) != 1),
            ValueError,
            "x and f must be arrays of equal length",
        )
        errorif(method not in METHODS, ValueError, f"unknown method {method}")

        self.x = x
        self.f = f
        self.axis = axis
        self.method = method
        self.extrap = extrap
        self.period = period  # pyright: ignore
        (self.even_spacing,) = _detect_even_spacing(even_spacing, (x,), (period,))

        if fx is None:
            fx = _approx_df(x, f, method, axis, period, **kwargs)

        self.derivs = {"fx": fx}

    def __call__(
        self, xq: Real[ArrayLike, "..."], dx: int = 0
    ) -> Inexact[Array, "..."]:
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
            self.even_spacing,
            axis=self.axis,
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
    even_spacing : bool, None, tuple
        Whether the knots along each axis are evenly spaced, which allows a faster
        lookup of the cell containing each query point. If None, this is checked
        once at construction, or on each call if constructed inside a jitted
        function. True skips the check, and gives incorrect results if the knots are
        not evenly spaced. False always uses a binary search. Defaults to None. Use a
        single value for the same in all directions. Along periodic axes, even
        spacing requires the knots to cover one full period without the endpoint,
        e.g. ``np.linspace(0, period, N, endpoint=False)``.
    fx, fy, fxy : ndarray, optional
        derivatives df/dx, df/dy, d2f/dxdy at the knots. For cubic methods, any that
        are given are used in place of the derivatives approximated by ``method``.
    axis : tuple of int, shape(2,), optional
        axes of f along which x and y vary. Default is (0, 1). The output has the
        remaining axes of f in their original order, with the query shape inserted at
        position ``min(axis)``.

    """

    x: Float[Array, " Nx"]
    y: Float[Array, " Ny"]
    f: Inexact[Array, " Nx Ny ..."]
    _fstack: Inexact[Array, " Nx Ny 4 ..."]
    method: str = eqx.field(static=True)
    extrap: bool | float | tuple
    period: None | float | tuple
    axis: int | tuple[int, ...] = eqx.field(static=True)
    even_spacing: bool | None | tuple = eqx.field(static=True)

    def __init__(
        self,
        x: Real[ArrayLike, " Nx"],
        y: Real[ArrayLike, " Ny"],
        f: Num[ArrayLike, " Nx Ny ..."],
        method: str = "cubic",
        extrap: bool | float | tuple = False,
        period: None | float | tuple = None,
        even_spacing: bool | None | tuple = None,
        **kwargs,
    ):
        x, y, f = map(asarray_inexact, (x, y, f))
        axis = _parse_axis(kwargs.pop("axis", None), 2, f.ndim)
        derivs = {name: kwargs.pop(name, None) for name in _field_names(2)[1:]}

        errorif(
            (len(x) != f.shape[axis[0]]) or (x.ndim != 1),
            ValueError,
            "x and f must be arrays of equal length",
        )
        errorif(
            (len(y) != f.shape[axis[1]]) or (y.ndim != 1),
            ValueError,
            "y and f must be arrays of equal length",
        )
        errorif(method not in METHODS, ValueError, f"unknown method {method}")

        self.x = x
        self.y = y
        self.f = f
        self.axis = axis
        self.method = method
        self.extrap = extrap
        self.period = period
        periodx, periody = _parse_ndarg(period, 2)
        self.even_spacing = _detect_even_spacing(
            even_spacing, (x, y), (periodx, periody)
        )

        self._fstack = _stack_fields(
            jnp.moveaxis(f, axis, tuple(range(len(axis)))),
            *_move_grid_axes(tuple(derivs.values()), axis),
            grids=(x, y),
            method=method,
            periods=(periodx, periody),
            **kwargs,
        )

    # equinox allows a property to implement an AbstractVar, which pyright doesn't model
    @property
    def derivs(self) -> dict[str, Inexact[Array, "..."]]:  # pyright: ignore[reportIncompatibleVariableOverride]
        """dict: Derivatives of f at the knots, keyed by ``fx``, ``fy``, ``fxy``."""
        assert isinstance(self.axis, tuple)
        return _unstack_fields(self._fstack, self.axis)

    def __call__(
        self,
        xq: Real[ArrayLike, "..."],
        yq: Real[ArrayLike, "..."],
        dx: int = 0,
        dy: int = 0,
    ) -> Inexact[Array, "..."]:
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
            self.even_spacing,
            axis=self.axis,
            **self._fstack_kwargs(),
        )

    def _fstack_kwargs(self) -> dict:
        # only cubic methods use the derivatives
        return {"_fstack": self._fstack} if self.method in CUBIC_METHODS else {}


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
    even_spacing : bool, None, tuple
        Whether the knots along each axis are evenly spaced, which allows a faster
        lookup of the cell containing each query point. If None, this is checked
        once at construction, or on each call if constructed inside a jitted
        function. True skips the check, and gives incorrect results if the knots are
        not evenly spaced. False always uses a binary search. Defaults to None. Use a
        single value for the same in all directions. Along periodic axes, even
        spacing requires the knots to cover one full period without the endpoint,
        e.g. ``np.linspace(0, period, N, endpoint=False)``.
    fx, fy, fz, fxy, fxz, fyz, fxyz : ndarray, optional
        derivatives df/dx, df/dy, df/dz, d2f/dxdy, d2f/dxdz, d2f/dydz, d3f/dxdydz at
        the knots. For cubic methods, any that are given are used in place of the
        derivatives approximated by ``method``.
    axis : tuple of int, shape(3,), optional
        axes of f along which x, y and z vary. Default is (0, 1, 2). The output has the
        remaining axes of f in their original order, with the query shape inserted at
        position ``min(axis)``.

    """

    x: Float[Array, " Nx"]
    y: Float[Array, " Ny"]
    z: Float[Array, " Nz"]
    f: Inexact[Array, " Nx Ny Nz ..."]
    _fstack: Inexact[Array, " Nx Ny Nz 8 ..."]
    method: str = eqx.field(static=True)
    extrap: bool | float | tuple
    period: None | float | tuple
    axis: int | tuple[int, ...] = eqx.field(static=True)
    even_spacing: bool | None | tuple = eqx.field(static=True)

    def __init__(
        self,
        x: Real[ArrayLike, " Nx"],
        y: Real[ArrayLike, " Ny"],
        z: Real[ArrayLike, " Nz"],
        f: Num[ArrayLike, " Nx Ny Nz ..."],
        method: str = "cubic",
        extrap: bool | float | tuple = False,
        period: None | float | tuple = None,
        even_spacing: bool | None | tuple = None,
        **kwargs,
    ):
        x, y, z, f = map(asarray_inexact, (x, y, z, f))
        axis = _parse_axis(kwargs.pop("axis", None), 3, f.ndim)

        errorif(
            (len(x) != f.shape[axis[0]]) or (x.ndim != 1),
            ValueError,
            "x and f must be arrays of equal length",
        )
        errorif(
            (len(y) != f.shape[axis[1]]) or (y.ndim != 1),
            ValueError,
            "y and f must be arrays of equal length",
        )
        errorif(
            (len(z) != f.shape[axis[2]]) or (z.ndim != 1),
            ValueError,
            "z and f must be arrays of equal length",
        )
        errorif(method not in METHODS, ValueError, f"unknown method {method}")

        derivs = {name: kwargs.pop(name, None) for name in _field_names(3)[1:]}

        self.x = x
        self.y = y
        self.z = z
        self.f = f
        self.axis = axis
        self.method = method
        self.extrap = extrap
        self.period = period
        periodx, periody, periodz = _parse_ndarg(period, 3)
        self.even_spacing = _detect_even_spacing(
            even_spacing, (x, y, z), (periodx, periody, periodz)
        )

        self._fstack = _stack_fields(
            jnp.moveaxis(f, axis, tuple(range(len(axis)))),
            *_move_grid_axes(tuple(derivs.values()), axis),
            grids=(x, y, z),
            method=method,
            periods=(periodx, periody, periodz),
            **kwargs,
        )

    # equinox allows a property to implement an AbstractVar, which pyright doesn't model
    @property
    def derivs(self) -> dict[str, Inexact[Array, "..."]]:  # pyright: ignore[reportIncompatibleVariableOverride]
        """dict: Derivatives of f at the knots, keyed by ``fx``, ``fy``, etc."""
        assert isinstance(self.axis, tuple)
        return _unstack_fields(self._fstack, self.axis)

    def _fstack_kwargs(self) -> dict:
        # only cubic methods use the derivatives
        return {"_fstack": self._fstack} if self.method in CUBIC_METHODS else {}

    def __call__(
        self,
        xq: Real[ArrayLike, "..."],
        yq: Real[ArrayLike, "..."],
        zq: Real[ArrayLike, "..."],
        dx: int = 0,
        dy: int = 0,
        dz: int = 0,
    ) -> Inexact[Array, "..."]:
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
            self.even_spacing,
            axis=self.axis,
            **self._fstack_kwargs(),
        )


@wrap_jit(static_argnames=["method", "axis", "even_spacing"])
def interp1d(
    xq: Real[ArrayLike, " Nq"],
    x: Real[ArrayLike, " Nx"],
    f: Num[ArrayLike, "Nx ..."],
    method: str = "cubic",
    derivative: int = 0,
    extrap: bool | float | tuple = False,
    period: None | float = None,
    even_spacing: bool | None = False,
    **kwargs,
) -> Inexact[Array, "Nq ..."]:
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
    even_spacing : bool, None
        Whether x is evenly spaced, which allows a faster lookup of the interval
        containing each query point. If None, this is checked on each call. True skips
        the check, and gives incorrect results if x is not evenly spaced. False always
        uses a binary search. Defaults to False. If periodic, even spacing requires x to
        cover one full period without the endpoint, e.g.
        ``np.linspace(0, period, N, endpoint=False)``.
    fx : ndarray, optional
        first derivatives of f with respect to x at the knots. For cubic methods, if
        given these are used in place of the derivatives approximated by ``method``.
    axis : int, optional
        axis of f along which x varies. Default is 0. The output has the query shape
        in place of this axis, ie
        ``fq.shape == f.shape[:axis] + xq.shape + f.shape[axis + 1 :]``.

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
    xtype = jnp.result_type(x, xq)
    x, xq = map(lambda a: a.astype(xtype), (x, xq))
    f = f.astype(jnp.result_type(f, x))
    axis = _parse_axis(kwargs.pop("axis", None), 1, f.ndim)
    fx = kwargs.pop("fx", None)
    # interpolate along the leading axis, then move the query axes into place
    f = jnp.moveaxis(f, axis, 0)
    if fx is not None:
        fx = jnp.moveaxis(fx, axis, 0)
    outshape = xq.shape + f.shape[1:]
    nq = xq.ndim

    # Promote scalar query points to 1D array.
    # Note this is done after the computation of outshape
    # to make jax.grad work in the scalar case.
    xq = jnp.atleast_1d(xq)

    errorif(
        (len(x) != f.shape[0]) or (jnp.ndim(x) != 1),
        ValueError,
        "x and f must be arrays of equal length",
    )
    errorif(method not in METHODS, ValueError, f"unknown method {method}")

    lowx, highx = _parse_extrap(extrap, 1)

    if method in CUBIC_METHODS and fx is None:
        fx = _approx_df(x, f, method, 0, period, **kwargs)

    if period is not None:
        xq, x, f, fx = _make_periodic(xq, x, period, 0, f, fx)
        lowx = highx = True

    i = _find_cell(x, xq, even_spacing)

    if method == "nearest":

        def derivative0_nearest():
            x0, x1 = x[i - 1], x[i]
            return f[jnp.where(xq - x0 <= x1 - xq, i - 1, i)]

        def derivative1_nearest():
            return jnp.zeros((xq.size, *f.shape[1:]), dtype=f.dtype)

        fq = jax.lax.switch(derivative, [derivative0_nearest, derivative1_nearest])

    elif method == "linear":

        def derivative0_linear():
            f0, f1 = f[i - 1], f[i]
            x0, x1 = x[i - 1], x[i]
            df = f1 - f0
            dx = x1 - x0
            dxi = jnp.where(dx == 0, 0, 1 / dx)
            delta = xq - x0
            fq = jnp.where(
                (dx == 0),
                f1.T,
                f0.T + (delta * dxi * df.T),
            ).T
            return fq

        def derivative1_linear():
            f0, f1 = f[i - 1], f[i]
            x0, x1 = x[i - 1], x[i]
            df = f1 - f0
            dx = x1 - x0
            dxi = jnp.where(dx == 0, 0, 1 / dx)
            return (df.T * dxi).T

        def derivative2_linear():
            return jnp.zeros((xq.size, *f.shape[1:]), dtype=f.dtype)

        fq = jax.lax.switch(
            derivative, [derivative0_linear, derivative1_linear, derivative2_linear]
        )

    else:
        assert method in CUBIC_METHODS
        assert fx is not None and fx.shape == f.shape

        x0, x1 = x[i - 1], x[i]
        dx = x1 - x0
        delta = xq - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        t = delta * dxi

        f0, f1 = f[i - 1], f[i]
        fx0, fx1 = fx[i - 1], fx[i]
        fx0 = (fx0.T * dx).T
        fx1 = (fx1.T * dx).T

        F = jnp.stack([f0, f1, fx0, fx1], axis=0).T
        # full precision, as the default for float32 on some GPUs keeps only ~10 bits
        matmul = functools.partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)
        coef = jnp.vectorize(matmul, signature="(n,n),(n)->(n)")(A_CUBIC, F).T
        ttx = _get_t_der(t, derivative, dxi)
        fq = jnp.einsum("ji...,ij->i...", coef, ttx)

    fq = _extrap(xq, fq, x, lowx, highx)
    return _move_query_axes(fq.reshape(outshape), nq, axis)


@wrap_jit(static_argnames=["method", "axis", "even_spacing"])
def interp2d(  # noqa: C901 - FIXME: break this up into simpler pieces
    xq: Real[ArrayLike, " Nq"],
    yq: Real[ArrayLike, " Nq"],
    x: Real[ArrayLike, " Nx"],
    y: Real[ArrayLike, " Ny"],
    f: Num[ArrayLike, "Nx Ny ..."],
    method: str = "cubic",
    derivative: int | tuple = 0,
    extrap: bool | float | tuple = False,
    period: None | float | tuple = None,
    even_spacing: bool | None | tuple = False,
    **kwargs,
) -> Inexact[Array, "Nq ..."]:
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
    even_spacing : bool, None, tuple
        Whether the knots along each axis are evenly spaced, which allows a faster
        lookup of the cell containing each query point. If None, this is checked on
        each call. True skips the check, and gives incorrect results if the knots are
        not evenly spaced. False always uses a binary search. Defaults to False. Use a
        single value for the same in all directions. Along periodic axes, even
        spacing requires the knots to cover one full period without the endpoint,
        e.g. ``np.linspace(0, period, N, endpoint=False)``.
    fx, fy, fxy : ndarray, optional
        derivatives df/dx, df/dy, d2f/dxdy at the knots. For cubic methods, any that
        are given are used in place of the derivatives approximated by ``method``.
    axis : tuple of int, shape(2,), optional
        axes of f along which x and y vary. Default is (0, 1). The output has the
        remaining axes of f in their original order, with the query shape inserted at
        position ``min(axis)``.

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
    xtype = jnp.result_type(x, y, xq, yq)
    x, y, xq, yq = map(lambda a: a.astype(xtype), (x, y, xq, yq))
    f = f.astype(jnp.result_type(f, x))

    axis = _parse_axis(kwargs.pop("axis", None), 2, f.ndim)
    derivs = {name: kwargs.pop(name, None) for name in _field_names(2)[1:]}
    # f and its derivatives stacked as by _stack_fields, as precomputed by
    # Interpolator2D
    fstack = kwargs.pop("_fstack", None)
    # interpolate along the leading axes, then move the query axes into place
    f = jnp.moveaxis(f, axis, tuple(range(len(axis))))
    xq, yq = jnp.broadcast_arrays(xq, yq)
    outshape = xq.shape + f.shape[2:]
    nq = xq.ndim

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
    errorif(method not in METHODS, ValueError, f"unknown method {method}")

    periodx, periody = _parse_ndarg(period, 2)
    derivative_x, derivative_y = _parse_ndarg(derivative, 2)
    lowx, highx, lowy, highy = _parse_extrap(extrap, 2)

    if method in CUBIC_METHODS and fstack is None:
        fstack = _stack_fields(
            f,
            *_move_grid_axes(tuple(derivs.values()), axis),
            grids=(x, y),
            method=method,
            periods=(periodx, periody),
            **kwargs,
        )
    # the values the method reads: the stack for cubic methods, otherwise just f
    data = fstack if method in CUBIC_METHODS else f

    if periodx is not None:
        xq, x, data = _make_periodic(xq, x, periodx, 0, data)
        lowx = highx = True
    if periody is not None:
        yq, y, data = _make_periodic(yq, y, periody, 1, data)
        lowy = highy = True

    evenx, eveny = _parse_ndarg(even_spacing, 2)
    i = _find_cell(x, xq, evenx)
    j = _find_cell(y, yq, eveny)

    if method == "nearest":

        def derivative0():
            # because of the regular spaced grid we know that the nearest point
            # will be one of the 4 neighbors on the grid, so we first find those
            # and then take the nearest one among them.
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[j - 1], y[j]
            F = _take_neighbor_data(data, i, j)
            neighbors_x = jnp.array([[x1, x0, x1, x0], [y1, y1, y0, y0]])
            neighbors_f = jnp.array([F[1, 1].T, F[0, 1].T, F[1, 0].T, F[0, 0].T])
            xyq = jnp.array([xq, yq])
            dist = jnp.linalg.norm(neighbors_x - xyq[:, None, :], axis=0)
            idx = jnp.argmin(dist, axis=0)
            return jax.vmap(lambda a, b: jnp.take(a, b, axis=-1))(neighbors_f.T, idx)

        def derivative1():
            return jnp.zeros((xq.size, *f.shape[2:]), dtype=f.dtype)

        fq = jax.lax.cond(
            (derivative_x == 0) & (derivative_y == 0), derivative0, derivative1
        )

    elif method == "linear":
        F = _take_neighbor_data(data, i, j)
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[j - 1], y[j]
        dx = x1 - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        dy = y1 - y0
        dyi = jnp.where(dy == 0, 0, 1 / dy)

        dx0 = lambda: jnp.array([x1 - xq, xq - x0])
        dx1 = lambda: jnp.array([-jnp.ones_like(xq), jnp.ones_like(xq)])
        dx2 = lambda: jnp.zeros((2, xq.size), dtype=xtype)
        dy0 = lambda: jnp.array([y1 - yq, yq - y0])
        dy1 = lambda: jnp.array([-jnp.ones_like(yq), jnp.ones_like(yq)])
        dy2 = lambda: jnp.zeros((2, yq.size), dtype=xtype)

        tx = jax.lax.switch(derivative_x, [dx0, dx1, dx2])
        ty = jax.lax.switch(derivative_y, [dy0, dy1, dy2])
        fq = (dxi * dyi * jnp.einsum("ijk...,ik,jk->k...", F, tx, ty).T).T

    else:
        assert method in CUBIC_METHODS
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[j - 1], y[j]
        dx = x1 - x0
        deltax = xq - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        tx = deltax * dxi
        dy = y1 - y0
        deltay = yq - y0
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        ty = deltay * dyi

        coef = _cell_coefficients(_take_neighbor_data(data, i, j), (dx, dy), A_BICUBIC)
        ttx = _get_t_der(tx, derivative_x, dxi)
        tty = _get_t_der(ty, derivative_y, dyi)
        fq = jnp.einsum("ijk...,ij,ik->i...", coef, ttx, tty)

    fq = _extrap(xq, fq, x, lowx, highx)
    fq = _extrap(yq, fq, y, lowy, highy)

    return _move_query_axes(fq.reshape(outshape), nq, axis)


@wrap_jit(static_argnames=["method", "axis", "even_spacing"])
def interp3d(  # noqa: C901 - FIXME: break this up into simpler pieces
    xq: Real[ArrayLike, " Nq"],
    yq: Real[ArrayLike, " Nq"],
    zq: Real[ArrayLike, " Nq"],
    x: Real[ArrayLike, " Nx"],
    y: Real[ArrayLike, " Ny"],
    z: Real[ArrayLike, " Nz"],
    f: Num[ArrayLike, "Nx Ny Nz ..."],
    method: str = "cubic",
    derivative: int | tuple = 0,
    extrap: bool | float | tuple = False,
    period: None | float | tuple = None,
    even_spacing: bool | None | tuple = False,
    **kwargs,
) -> Inexact[Array, "Nq ..."]:
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
    even_spacing : bool, None, tuple
        Whether the knots along each axis are evenly spaced, which allows a faster
        lookup of the cell containing each query point. If None, this is checked on
        each call. True skips the check, and gives incorrect results if the knots are
        not evenly spaced. False always uses a binary search. Defaults to False. Use a
        single value for the same in all directions. Along periodic axes, even
        spacing requires the knots to cover one full period without the endpoint,
        e.g. ``np.linspace(0, period, N, endpoint=False)``.
    fx, fy, fz, fxy, fxz, fyz, fxyz : ndarray, optional
        derivatives df/dx, df/dy, df/dz, d2f/dxdy, d2f/dxdz, d2f/dydz, d3f/dxdydz at
        the knots. For cubic methods, any that are given are used in place of the
        derivatives approximated by ``method``.
    axis : tuple of int, shape(3,), optional
        axes of f along which x, y and z vary. Default is (0, 1, 2). The output has the
        remaining axes of f in their original order, with the query shape inserted at
        position ``min(axis)``.

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
    xtype = jnp.result_type(x, y, z, xq, yq, zq)
    x, y, z, xq, yq, zq = map(lambda a: a.astype(xtype), (x, y, z, xq, yq, zq))
    f = f.astype(jnp.result_type(f, x))

    axis = _parse_axis(kwargs.pop("axis", None), 3, f.ndim)
    derivs = {name: kwargs.pop(name, None) for name in _field_names(3)[1:]}
    # f and its derivatives stacked as by _stack_fields, as precomputed by
    # Interpolator3D
    fstack = kwargs.pop("_fstack", None)
    # interpolate along the leading axes, then move the query axes into place
    f = jnp.moveaxis(f, axis, tuple(range(len(axis))))

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
    errorif(method not in METHODS, ValueError, f"unknown method {method}")

    xq, yq, zq = jnp.broadcast_arrays(xq, yq, zq)
    outshape = xq.shape + f.shape[3:]
    nq = xq.ndim

    # Promote scalar query points to 1D array.
    # Note this is done after the computation of outshape
    # to make jax.grad work in the scalar case.
    xq, yq, zq = map(jnp.atleast_1d, (xq, yq, zq))

    periodx, periody, periodz = _parse_ndarg(period, 3)
    derivative_x, derivative_y, derivative_z = _parse_ndarg(derivative, 3)
    lowx, highx, lowy, highy, lowz, highz = _parse_extrap(extrap, 3)

    if method in CUBIC_METHODS and fstack is None:
        fstack = _stack_fields(
            f,
            *_move_grid_axes(tuple(derivs.values()), axis),
            grids=(x, y, z),
            method=method,
            periods=(periodx, periody, periodz),
            **kwargs,
        )
    # the values the method reads: the stack for cubic methods, otherwise just f
    data = fstack if method in CUBIC_METHODS else f

    if periodx is not None:
        xq, x, data = _make_periodic(xq, x, periodx, 0, data)
        lowx = highx = True
    if periody is not None:
        yq, y, data = _make_periodic(yq, y, periody, 1, data)
        lowy = highy = True
    if periodz is not None:
        zq, z, data = _make_periodic(zq, z, periodz, 2, data)
        lowz = highz = True

    evenx, eveny, evenz = _parse_ndarg(even_spacing, 3)
    i = _find_cell(x, xq, evenx)
    j = _find_cell(y, yq, eveny)
    k = _find_cell(z, zq, evenz)

    if method == "nearest":

        def derivative0():
            # because of the regular spaced grid we know that the nearest point
            # will be one of the 8 neighbors on the grid, so we first find those
            # and then take the nearest one among them.
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[j - 1], y[j]
            z0, z1 = z[k - 1], z[k]
            F = _take_neighbor_data(data, i, j, k)
            neighbors_x = jnp.array(
                [
                    [x1, x0, x1, x0, x1, x0, x1, x0],
                    [y1, y1, y0, y0, y1, y1, y0, y0],
                    [z1, z1, z1, z1, z0, z0, z0, z0],
                ]
            )
            neighbors_f = jnp.array(
                [
                    F[1, 1, 1].T,
                    F[0, 1, 1].T,
                    F[1, 0, 1].T,
                    F[0, 0, 1].T,
                    F[1, 1, 0].T,
                    F[0, 1, 0].T,
                    F[1, 0, 0].T,
                    F[0, 0, 0].T,
                ]
            )
            xyzq = jnp.array([xq, yq, zq])
            dist = jnp.linalg.norm(neighbors_x - xyzq[:, None, :], axis=0)
            idx = jnp.argmin(dist, axis=0)
            return jax.vmap(lambda a, b: jnp.take(a, b, axis=-1))(neighbors_f.T, idx)

        def derivative1():
            return jnp.zeros((xq.size, *f.shape[3:]), dtype=f.dtype)

        fq = jax.lax.cond(
            (derivative_x == 0) & (derivative_y == 0) & (derivative_z == 0),
            derivative0,
            derivative1,
        )

    elif method == "linear":
        F = _take_neighbor_data(data, i, j, k)
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[j - 1], y[j]
        z0, z1 = z[k - 1], z[k]
        dx = x1 - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        dy = y1 - y0
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        dz = z1 - z0
        dzi = jnp.where(dz == 0, 0, 1 / dz)

        dx0 = lambda: jnp.array([x1 - xq, xq - x0])
        dx1 = lambda: jnp.array([-jnp.ones_like(xq), jnp.ones_like(xq)])
        dx2 = lambda: jnp.zeros((2, xq.size), dtype=xtype)
        dy0 = lambda: jnp.array([y1 - yq, yq - y0])
        dy1 = lambda: jnp.array([-jnp.ones_like(yq), jnp.ones_like(yq)])
        dy2 = lambda: jnp.zeros((2, yq.size), dtype=xtype)
        dz0 = lambda: jnp.array([z1 - zq, zq - z0])
        dz1 = lambda: jnp.array([-jnp.ones_like(zq), jnp.ones_like(zq)])
        dz2 = lambda: jnp.zeros((2, zq.size), dtype=xtype)

        tx = jax.lax.switch(derivative_x, [dx0, dx1, dx2])
        ty = jax.lax.switch(derivative_y, [dy0, dy1, dy2])
        tz = jax.lax.switch(derivative_z, [dz0, dz1, dz2])

        fq = (dxi * dyi * dzi * jnp.einsum("lijk...,lk,ik,jk->k...", F, tx, ty, tz).T).T

    else:
        assert method in CUBIC_METHODS
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[j - 1], y[j]
        z0, z1 = z[k - 1], z[k]

        dx = x1 - x0
        deltax = xq - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        tx = deltax * dxi

        dy = y1 - y0
        deltay = yq - y0
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        ty = deltay * dyi

        dz = z1 - z0
        deltaz = zq - z0
        dzi = jnp.where(dz == 0, 0, 1 / dz)
        tz = deltaz * dzi

        coef = _cell_coefficients(
            _take_neighbor_data(data, i, j, k), (dx, dy, dz), A_TRICUBIC
        )
        ttx = _get_t_der(tx, derivative_x, dxi)
        tty = _get_t_der(ty, derivative_y, dyi)
        ttz = _get_t_der(tz, derivative_z, dzi)
        fq = jnp.einsum("lijk...,li,lj,lk->l...", coef, ttx, tty, ttz)

    fq = _extrap(xq, fq, x, lowx, highx)
    fq = _extrap(yq, fq, y, lowy, highy)
    fq = _extrap(zq, fq, z, lowz, highz)

    return _move_query_axes(fq.reshape(outshape), nq, axis)


def _parse_axis(
    axis: None | int | tuple[int, ...], n: int, ndim: int
) -> tuple[int, ...]:
    """Normalize the axes of an array of rank ndim that correspond to n grid dims."""
    if axis is None:
        axis = tuple(range(n))
    elif isinstance(axis, (int, np.integer)):
        errorif(n != 1, ValueError, f"axis must be a tuple of {n} ints, got {axis}")
        axis = (axis,)
    axis = tuple(int(a) for a in axis)
    errorif(len(axis) != n, ValueError, f"axis must be a tuple of {n} ints, got {axis}")
    errorif(
        any(not (-ndim <= a < ndim) for a in axis),
        ValueError,
        f"axis {axis} out of bounds for array of dimension {ndim}",
    )
    axis = tuple(a % ndim for a in axis)
    errorif(len(set(axis)) != n, ValueError, f"repeated axis in {axis}")
    return axis


def _move_query_axes(fq: jax.Array, nq: int, axis: tuple[int, ...]) -> jax.Array:
    """Move leading nq query dimensions to where the grid axes of f started."""
    start = min(axis)
    return jnp.moveaxis(fq, tuple(range(nq)), tuple(range(start, start + nq)))


@jit
def _is_uniform(x: jax.Array) -> jax.Array:
    """Whether the knots x are close enough to evenly spaced for a direct lookup."""
    n = len(x)
    dx = (x[-1] - x[0]) / (n - 1)
    dev = jnp.max(jnp.abs(x - (x[0] + dx * jnp.arange(n, dtype=x.dtype))))
    # every knot within a quarter spacing of the uniform grid means the direct guess
    # is at most one cell off, even allowing for rounding in computing the guess, so
    # a single correction step makes it exact.
    return dev < dx / 4


def _find_cell_uniform(x: jax.Array, xq: jax.Array) -> jax.Array:
    n = len(x)
    # spacing from the full extent rather than the first interval, so that rounding
    # error in the spacing isn't amplified by the knot index.
    dx = (x[-1] - x[0]) / (n - 1)
    i = jnp.clip(jnp.floor((xq - x[0]) / dx).astype(int) + 1, 1, n - 1)
    i = i + (x[i] <= xq) - (x[i - 1] > xq)
    return jnp.clip(i, 1, n - 1)


def _find_cell_search(x: jax.Array, xq: jax.Array) -> jax.Array:
    # unrolling the binary search lets XLA fuse its steps, which is much faster than
    # the default loop on accelerators.
    i = jnp.searchsorted(x, xq, side="right", method="scan_unrolled")
    return jnp.clip(i.astype(int), 1, len(x) - 1)


def _find_cell(x: jax.Array, xq: jax.Array, even_spacing: bool | None) -> jax.Array:
    """Find the index i such that x[i - 1] <= xq < x[i], clipped to [1, len(x) - 1].

    Parameters
    ----------
    x : ndarray, shape(N,)
        Monotonically increasing knots.
    xq : ndarray, shape(Nq,)
        Query points.
    even_spacing : bool, None
        Whether x is evenly spaced, in which case the index is computed directly
        instead of by binary search. If None, this is checked at runtime.

    Returns
    -------
    i : ndarray of int, shape(Nq,)
        Index of the upper knot of the cell containing each query point.
    """
    if even_spacing is None:
        return jax.lax.cond(
            _is_uniform(x), _find_cell_uniform, _find_cell_search, x, xq
        )
    if even_spacing:
        return _find_cell_uniform(x, xq)
    return _find_cell_search(x, xq)


def _detect_even_spacing(
    even_spacing: bool | None | tuple, grids: tuple, periods: tuple
) -> tuple:
    """Resolve even_spacing per axis, checking concrete grids where it is None."""
    out = []
    for even, x, period in zip(_parse_ndarg(even_spacing, len(grids)), grids, periods):
        if even is None:
            if period is not None:
                _, x = _make_periodic(x, x, period, 0)
            try:
                even = bool(_is_uniform(x))
            except jax.errors.ConcretizationTypeError:
                pass  # traced grid, so leave the check to each call
        out.append(even)
    return tuple(out)


def _field_names(n: int) -> tuple[str, ...]:
    """Names of f and its derivatives used for cubic interpolation in n dimensions.

    These are f and every mixed first derivative, ordered by number of derivatives
    and then by axis, e.g. f, fx, fy, fxy in 2D.
    """
    return tuple(
        "f" + "".join(c)
        for r in range(n + 1)
        for c in itertools.combinations("xyz"[:n], r)
    )


def _move_grid_axes(
    arrs: tuple[jax.Array | None, ...], axis: tuple[int, ...]
) -> tuple[jax.Array | None, ...]:
    """Move the grid axes of each array (skipping None) to the front."""
    lead = tuple(range(len(axis)))
    return tuple(None if a is None else jnp.moveaxis(a, axis, lead) for a in arrs)


@eqx.filter_jit
def _stack_fields(
    f: jax.Array,
    *derivs: jax.Array | None,
    grids: tuple,
    method: str,
    periods: tuple,
    **kwargs,
) -> jax.Array:
    """Stack f and its derivatives along a new axis right after the grid axes.

    Parameters
    ----------
    f : ndarray, shape(N1, ..., Nn, ...)
        Function values, with the grid axes leading.
    *derivs : ndarray or None
        Derivatives of f in the order of _field_names(n)[1:], with the grid axes
        leading. Any that are None are approximated with ``method``.
    grids : tuple of ndarray
        Knots along each grid axis.
    method : str
        Method used to approximate missing derivatives.
    periods : tuple of float or None
        Period along each grid axis.

    Returns
    -------
    fstack : ndarray, shape(N1, ..., Nn, 2**n, ...)
        f and its derivatives, in the order of _field_names(n).
    """
    n = len(grids)
    names = _field_names(n)
    fields = {"f": f, **dict(zip(names[1:], derivs))}
    for name in names[1:]:
        if fields[name] is None:
            # derivative along the last axis in the name, of the field before it
            k = "xyz".index(name[-1])
            fields[name] = _approx_df(
                grids[k], fields[name[:-1]], method, k, periods[k], **kwargs
            )
    # with the fields interleaved, the values of every field at a cell corner are
    # adjacent in memory, so cubic evaluation reads them with a single gather
    return jnp.stack([fields[name] for name in names], axis=n)


def _unstack_fields(fstack: jax.Array, axis: tuple[int, ...]) -> dict:
    """Derivatives from _stack_fields, with the grid axes moved back to axis."""
    n = len(axis)
    return {
        name: jnp.moveaxis(jnp.take(fstack, m, axis=n), tuple(range(n)), axis)
        for m, name in enumerate(_field_names(n))
        if m > 0
    }


def _cell_coefficients(
    corners: jax.Array, widths: tuple[jax.Array, ...], A: np.ndarray
) -> jax.Array:
    """Coefficients of the cubic polynomial on each cell, from its corner values.

    Parameters
    ----------
    corners : ndarray, shape(2, ..., 2, Nq, 2**n, ...)
        Values of f and its derivatives (in the order of _field_names(n)) at each
        cell corner, as gathered from the output of _stack_fields.
    widths : tuple of ndarray, shape(Nq,)
        Width of the cell along each axis.
    A : ndarray, shape(4**n, 4**n)
        Matrix mapping corner values to polynomial coefficients.

    Returns
    -------
    coef : ndarray, shape(Nq, 4, ..., 4, ...)
        Coefficient of t1**p1 * ... * tn**pn at index [:, p1, ..., pn].
    """
    n = len(widths)
    nq = corners.shape[n]
    trailing = corners.shape[n + 2 :]
    # derivatives with respect to the normalized cell coordinates, which are scaled
    # by the cell width along each differentiated axis
    width = dict(zip("xyz", widths))

    def field_scale(name):
        out = jnp.ones_like(widths[0])
        for a in name[1:]:
            out = out * width[a]
        return out

    scale = jnp.stack([field_scale(name) for name in _field_names(n)], axis=-1)
    corners = corners * scale.reshape(scale.shape + (1,) * len(trailing))
    # flatten to the order A expects: field, then corner along the last axis down to
    # the first, i.e. (Nq, field, cn, ..., c1)
    corners = jnp.moveaxis(
        corners, (n, n + 1, *range(n - 1, -1, -1)), tuple(range(n + 2))
    )
    # full precision, as the default for float32 on some GPUs keeps only ~10 bits
    coef = jnp.einsum(
        "ij,qj...->qi...",
        A,
        corners.reshape((nq, -1, *trailing)),
        precision=jax.lax.Precision.HIGHEST,
    )
    # coefficient index is p1 + 4 * p2 + ..., so reshape gives (Nq, pn, ..., p1)
    coef = coef.reshape((nq,) + (4,) * n + trailing)
    return jnp.moveaxis(coef, tuple(range(1, n + 1)), tuple(range(n, 0, -1)))


def _take_neighbor_data(f: jax.Array, *idx: jax.Array) -> jax.Array:
    """Get the values of f on the grid cell containing each query point.

    For n grid dimensions, each query point lies in a cell with 2**n corners
    (2 in 1D, 2x2 in 2D, 2x2x2 in 3D). This returns f at those corners.

    Parameters
    ----------
    f : ndarray, shape(N1, ..., Nn, ...)
        Values on the grid. The first n axes are the grid axes; any trailing axes
        (e.g. several fields stacked together) are returned whole.
    *idx : ndarray, shape(Nq,)
        One index array per grid axis (i, j, k, ...). The cell of query point q
        spans [i[q] - 1, i[q]] along the first axis, [j[q] - 1, j[q]] along the
        second, and so on.

    Returns
    -------
    out : ndarray, shape(2, ..., 2, Nq, ...)
        n axes of size 2 for the cell corners, then the query axis, then the
        trailing axes of f. For example in 3D, out[a, b, c, q] is
        f[i[q] - 1 + a, j[q] - 1 + b, k[q] - 1 + c], so out[0, 0, 0] is the
        lower corner and out[1, 1, 1] the upper corner for every query point.

    Notes
    -----
    This uses one batched dynamic_slice instead of indexing f separately for each
    corner, which compiles to a single gather and is faster when f does not fit
    in cache.
    """
    n = len(idx)

    def take(*start):
        return jax.lax.dynamic_slice(
            f,
            # lower corner of the cell, and the start of any trailing axes
            tuple(s - 1 for s in start) + (0,) * (f.ndim - n),
            # 2 points along each grid axis, all of each trailing axis
            (2,) * n + f.shape[n:],
        )

    return jax.vmap(take, out_axes=n)(*idx)


@eqx.filter_jit
def _approx_df(
    x: jax.Array,
    f: jax.Array,
    method: str,
    axis: int,
    period: None | float,
    **kwargs,
) -> jax.Array:
    """Approximate df/dx along axis, treating f as periodic if period is given."""
    if period is None:
        return approx_df(x, f, method, axis, **kwargs)
    # Approximate on the periodically padded data so knots adjacent to the boundary
    # see their wrapped neighbors, then drop the padding and undo the sort so the
    # result lines up with f. Later padding copies these values, so every copy of a
    # knot has the same derivative.
    _, xp, fp = _make_periodic(x, x, period, axis, f)
    fx = approx_df(xp, fp, method, axis, **kwargs)
    fx = jnp.take(fx, jnp.arange(1, x.size + 1), axis)
    unsort = jnp.argsort(jnp.argsort(x % abs(period)))
    return jnp.take(fx, unsort, axis)


@wrap_jit(static_argnames=["axis"])
def _make_periodic(
    xq: jax.Array,
    x: jax.Array,
    period: float,
    axis: int,
    *arrs: jax.Array | None,
) -> tuple[jax.Array, ...]:
    """Make arrays periodic along a specified axis."""
    period = abs(period)
    xq = xq % period
    x = x % period
    i = jnp.argsort(x)
    x = x[i]
    x = jnp.concatenate([x[-1:] - period, x, x[:1] + period])
    arrlist: list[Any] = list(arrs)
    for k, arr in enumerate(arrlist):
        if arr is not None:
            arr = jnp.take(arr, i, axis, mode="wrap")
            arrlist[k] = jnp.concatenate(
                [
                    jnp.take(arr, jnp.array([-1]), axis),
                    arr,
                    jnp.take(arr, jnp.array([0]), axis),
                ],
                axis=axis,
            )
    return (xq, x, *arrlist)


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


def _parse_ndarg(arg: Any, n: int) -> Any | tuple:
    try:
        k = len(arg)
    except TypeError:
        arg = tuple(arg for _ in range(n))
        k = n
    errorif(k != n, ValueError, f"expected {n} values, got {k}")
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
    lo: bool | float,
    hi: bool | float,
):
    """Clamp or extrapolate values outside bounds."""

    def loclip(fq: jax.Array, lo: bool | float):
        # lo is either False (no extrapolation) or a fixed value to fill in
        if isbool(lo):
            lo = jnp.nan
        return jnp.where(xq < x[0], lo, fq.T).T

    def hiclip(fq: jax.Array, hi: bool | float):
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
