"""Functions for interpolating splines that are JAX differentiable."""

import numbers
from collections import OrderedDict

import jax.numpy as jnp
import numpy as np


def interp1d(
    xq, x, f, method="cubic", derivative=0, extrap=False, period=None, fx=None, **kwargs
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
        - `'nearest'`: nearest neighbor interpolation
        - `'linear'`: linear interpolation
        - `'cubic'`: C1 cubic splines (aka local splines)
        - `'cubic2'`: C2 cubic splines (aka natural splines)
        - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        - `'cardinal'`: C1 cubic general tension splines. If used, can also pass keyword
            parameter `c` in float[0,1] to specify tension
        - `'monotonic'`: C1 cubic splines that attempt to preserve monotonicity in the
            data, and will not introduce new extrema in the interpolated points
        - `'monotonic-0'`: same as `'monotonic'` but with 0 first derivatives at both
            endpoints
    derivative : int
        derivative order to calculate
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as a 2 element array or tuple to specify different conditions
        for xq<x[0] and x[-1]<xq
    period : float, None
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]
    fx : ndarray, shape(Nx,...)
        specified derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points
    """
    xq, x, f = map(jnp.asarray, (xq, x, f))
    axis = kwargs.get("axis", 0)
    lowx, highx = np.broadcast_to(extrap, (2,))

    if fx is not None:
        fx = jnp.asarray(fx)
    if len(x) != f.shape[axis] or jnp.ndim(x) != 1:
        raise ValueError("x and f must be arrays of equal length")
    if fx is not None and fx.shape != f.shape:
        raise ValueError(f"f and fx must have same shape, got {f.shape}, {fx.shape}")
    if derivative < 0:
        raise ValueError("derivative order should be non-negative")
    if period not in [0, None]:
        xq, x, f, fx = _make_periodic(xq, x, period, axis, f, fx)
        lowx = highx = True

    if method == "nearest":
        if derivative == 0:
            i = jnp.argmin(jnp.abs(xq[:, np.newaxis] - x[np.newaxis]), axis=1)
            fq = f[i]
        else:
            fq = jnp.zeros((xq.size, *f.shape[1:]))

    elif method == "linear":
        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        df = jnp.take(f, i, axis) - jnp.take(f, i - 1, axis)
        dx = x[i] - x[i - 1]
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        delta = xq - x[i - 1]
        if derivative == 0:
            fq = jnp.where(
                (dx == 0),
                jnp.take(f, i, axis),
                jnp.take(f, i - 1, axis) + delta * dxi * df,
            )
        elif derivative == 1:
            fq = df * dxi
        else:
            fq = jnp.zeros((xq.size, *f.shape[1:]))

    elif method in [
        "cubic",
        "cubic2",
        "cardinal",
        "catmull-rom",
        "monotonic",
        "monotonic-0",
    ]:
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

    else:
        raise ValueError(f"unknown method {method}")

    fq = _extrap(xq, fq, x, f, lowx, highx, axis)
    return fq


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
    fx=None,
    fy=None,
    fxy=None,
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
        - `'nearest'`: nearest neighbor interpolation
        - `'linear'`: linear interpolation
        - `'cubic'`: C1 cubic splines (aka local splines)
        - `'cubic2'`: C2 cubic splines (aka natural splines)
        - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        - `'cardinal'`: c1 cubic general tension splines. If used, can also pass keyword
            parameter `c` in float[0,1] to specify tension
    derivative : int, array-like
        derivative order to calculate, scalar values uses the same order for all
        coordinates, or pass a 2 element array or tuple to specify different derivatives
        in x,y directions
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions
        [[xlow, xhigh],[ylow,yhigh]]
    period : float, None, array-like
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. Pass a 2 element array or tuple to specify different
        periods for x and y coordinates
    fx : ndarray, shape(Nx,Ny,...)
        specified x derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fy : ndarray, shape(Nx,Ny,...)
        specified y derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fxy : ndarray, shape(Nx,Ny,...)
        specified mixed derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points
    """
    xq, yq, x, y, f = map(jnp.asarray, (xq, yq, x, y, f))
    period, extrap = map(np.asarray, (period, extrap))
    if len(x) != f.shape[0] or x.ndim != 1:
        raise ValueError("x and f must be arrays of equal length")
    if len(y) != f.shape[1] or y.ndim != 1:
        raise ValueError("y and f must be arrays of equal length")

    periodx, periody = np.broadcast_to(period, (2,))
    derivative_x, derivative_y = np.broadcast_to(derivative, (2,))
    lowx, highx, lowy, highy = np.broadcast_to(extrap, (2, 2)).flatten()

    if periodx not in [0, None]:
        xq, x, f, fx, fy, fxy = _make_periodic(xq, x, periodx, 0, f, fx, fy, fxy)
        lowx = highx = True
    if periody not in [0, None]:
        yq, y, f, fx, fy, fxy = _make_periodic(yq, y, periody, 1, f, fx, fy, fxy)
        lowy = highy = True

    if method == "nearest":
        if derivative_x in [0, None] and derivative_y in [0, None]:
            i = jnp.argmin(jnp.abs(xq[:, np.newaxis] - x[np.newaxis]), axis=1)
            j = jnp.argmin(jnp.abs(yq[:, np.newaxis] - y[np.newaxis]), axis=1)
            fq = f[i, j]
        else:
            fq = jnp.zeros((xq.size, yq.size, *f.shape[2:]))

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
        if derivative_x in [0, None]:
            tx = jnp.array([x1 - xq, xq - x0])
        elif derivative_x == 1:
            tx = jnp.array([-jnp.ones_like(xq), jnp.ones_like(xq)])
        else:
            tx = jnp.zeros((2, xq.size))
        if derivative_y in [0, None]:
            ty = jnp.array([y1 - yq, yq - y0])
        elif derivative_y == 1:
            ty = jnp.array([-jnp.ones_like(yq), jnp.ones_like(yq)])
        else:
            ty = jnp.zeros((2, yq.size))
        F = jnp.array([[f00, f01], [f10, f11]])
        fq = dxi * dyi * jnp.einsum("ik,ijk,jk->k", tx, F, ty)

    elif method in ["cubic", "cubic2", "cardinal", "catmull-rom"]:
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

    else:
        raise ValueError(f"unknown method {method}")

    fq = _extrap(xq, fq, x, f, lowx, highx, axis=0)
    fq = _extrap(yq, fq, y, f, lowy, highy, axis=1)

    return fq


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
    period=0,
    fx=None,
    fy=None,
    fz=None,
    fxy=None,
    fxz=None,
    fyz=None,
    fxyz=None,
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
        - `'nearest'`: nearest neighbor interpolation
        - `'linear'`: linear interpolation
        - `'cubic'`: C1 cubic splines (aka local splines)
        - `'cubic2'`: C2 cubic splines (aka natural splines)
        - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        - `'cardinal'`: c1 cubic general tension splines. If used, can also pass keyword
            parameter `c` in float[0,1] to specify tension
    derivative : int, array-like
        derivative order to calculate, scalar values uses the same order for all
        coordinates, or pass a 3 element array or tuple to specify different derivatives
        in x,y,z directions
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions for
        [[xlow, xhigh],[ylow,yhigh],[zlow,zhigh]]
    period : float, None, array-like
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. Pass a 3 element array or tuple to specify different
        periods for x,y,z coordinates
    fx : ndarray, shape(Nx,Ny,Nz,...)
        specified x derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fy : ndarray, shape(Nx,Ny,Nz,...)
        specified y derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fz : ndarray, shape(Nx,Ny,Nz,...)
        specified z derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fxy : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fxz : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fyz : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation
    fxyz : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated
        internally using `method`. Only used for cubic interpolation

    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points
    """
    xq, yq, zq, x, y, z, f = map(jnp.asarray, (xq, yq, zq, x, y, z, f))
    period, extrap = map(np.asarray, (period, extrap))
    if len(x) != f.shape[0] or x.ndim != 1:
        raise ValueError("x and f must be arrays of equal length")
    if len(y) != f.shape[1] or y.ndim != 1:
        raise ValueError("y and f must be arrays of equal length")
    if len(z) != f.shape[2] or z.ndim != 1:
        raise ValueError("z and f must be arrays of equal length")

    periodx, periody, periodz = np.broadcast_to(
        np.where(period == None, 0, period), (3,)  # noqa: E711
    )

    derivative_x, derivative_y, derivative_z = np.broadcast_to(
        np.where(derivative == None, 0, derivative), (3,)  # noqa: E711
    )
    lowx, highx, lowy, highy, lowz, highz = np.broadcast_to(extrap, (3, 2)).flatten()

    if periodx not in [0, None]:
        xq, x, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            xq, x, periodx, 0, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowx = highx = True
    if periody not in [0, None]:
        yq, y, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            yq, y, periody, 1, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowy = highy = True
    if periodz not in [0, None]:
        zq, z, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            zq, z, periodz, 2, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowz = highz = True

    if method == "nearest":
        if (
            derivative_x in [0, None]
            and derivative_y in [0, None]
            and derivative_z in [0, None]
        ):
            i = jnp.argmin(jnp.abs(xq[:, np.newaxis] - x[np.newaxis]), axis=1)
            j = jnp.argmin(jnp.abs(yq[:, np.newaxis] - y[np.newaxis]), axis=1)
            k = jnp.argmin(jnp.abs(zq[:, np.newaxis] - z[np.newaxis]), axis=1)
            fq = f[i, j, k]
        else:
            fq = jnp.zeros((xq.size, yq.size, zq.size, *f.shape[3:]))

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
        if derivative_x in [0, None]:
            tx = jnp.array([x1 - xq, xq - x0])
        elif derivative_x == 1:
            tx = jnp.array([-jnp.ones_like(xq), jnp.ones_like(xq)])
        else:
            tx = jnp.zeros((2, xq.size))
        if derivative_y in [0, None]:
            ty = jnp.array([y1 - yq, yq - y0])
        elif derivative_y == 1:
            ty = jnp.array([-jnp.ones_like(yq), jnp.ones_like(yq)])
        else:
            ty = jnp.zeros((2, yq.size))
        if derivative_z in [0, None]:
            tz = jnp.array([z1 - zq, zq - z0])
        elif derivative_z == 1:
            tz = jnp.array([-jnp.ones_like(zq), jnp.ones_like(zq)])
        else:
            tz = jnp.zeros((2, zq.size))
        F = jnp.array([[[f000, f010], [f100, f110]], [[f001, f011], [f101, f111]]])
        fq = dxi * dyi * dzi * jnp.einsum("il,ijkl,jl,kl->l", tx, F, ty, tz)

    elif method in ["cubic", "cubic2", "cardinal", "catmull-rom"]:
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

    else:
        raise ValueError(f"unknown method {method}")

    fq = _extrap(xq, fq, x, f, lowx, highx, axis=0)
    fq = _extrap(yq, fq, y, f, lowy, highy, axis=1)
    fq = _extrap(zq, fq, z, f, lowz, highz, axis=2)

    return fq


def _make_periodic(xq, x, period, axis, *arrs):
    """Make arrays periodic along a specified axis."""
    if period == 0:
        raise ValueError(f"period must be a non-zero value; got {period}")
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


def _get_t_der(t, derivative, dxi):
    """Get arrays of [1,t,t^2,t^3] for cubic interpolation."""
    assert int(derivative) == derivative, "derivative must be an integer"
    if derivative == 0 or derivative is None:
        tt = jnp.array([jnp.ones_like(t), t, t**2, t**3]).T
    elif derivative == 1:
        tt = (
            jnp.array([jnp.zeros_like(t), jnp.ones_like(t), 2 * t, 3 * t**2]).T
            * dxi[:, np.newaxis]
        )
    elif derivative == 2:
        tt = (
            jnp.array(
                [jnp.zeros_like(t), jnp.zeros_like(t), 2 * jnp.ones_like(t), 6 * t]
            ).T
            * dxi[:, np.newaxis] ** 2
        )
    elif derivative == 3:
        tt = (
            jnp.array(
                [
                    jnp.zeros_like(t),
                    jnp.zeros_like(t),
                    jnp.zeros_like(t),
                    6 * jnp.ones_like(t),
                ]
            ).T
            * dxi[:, np.newaxis] ** 3
        )
    else:
        tt = jnp.array(
            [jnp.zeros_like(t), jnp.zeros_like(t), jnp.zeros_like(t), jnp.zeros_like(t)]
        ).T
    return tt


def _extrap(xq, fq, x, f, low, high, axis=0):
    """Clamp or extrapolate values outside bounds."""
    if isinstance(low, numbers.Number) or (not low):
        low = low if isinstance(low, numbers.Number) else np.nan
        fq = jnp.where(xq < x[0], low, fq)
    if isinstance(high, numbers.Number) or (not high):
        high = high if isinstance(high, numbers.Number) else np.nan
        fq = jnp.where(xq > x[-1], high, fq)
    return fq


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
        fx0 = (
            (
                jnp.take(f, jnp.array([1]), axis, mode="wrap")
                - jnp.take(f, jnp.array([0]), axis, mode="wrap")
            )
            / (x[(1,)] - x[(0,)])
            if x[0] != x[1]
            else jnp.zeros_like(jnp.take(f, jnp.array([0]), axis, mode="wrap"))
        )
        fx1 = (
            (
                jnp.take(f, jnp.array([-1]), axis, mode="wrap")
                - jnp.take(f, jnp.array([-2]), axis, mode="wrap")
            )
            / (x[(-1,)] - x[(-2,)])
            if x[-1] != x[-2]
            else jnp.zeros_like(jnp.take(f, jnp.array([0]), axis, mode="wrap"))
        )
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

        dk = np.concatenate([d0, dk, d1])
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
