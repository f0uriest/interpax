"""Functions for interpolating splines that are JAX differentiable.

The docstrings and API are from SciPy, under a BSD license:

Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from functools import partial
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import jit

from ._coefs import A_CUBIC
from ._fd_derivs import approx_df
from .utils import asarray_inexact, errorif


class PPoly(eqx.Module):
    """Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.
    check : bool
        Whether to perform checks on the input. Should be False if used under JIT.

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.
    """

    _c: jax.Array
    _x: jax.Array
    _extrapolate: Union[bool, str] = eqx.field(static=True)
    _axis: int = eqx.field(static=True)

    def __init__(
        self,
        c: jax.Array,
        x: jax.Array,
        extrapolate: Union[bool, str] = None,
        axis: int = 0,
        check: bool = True,
    ):
        c = asarray_inexact(c)
        x = asarray_inexact(x)

        errorif(
            c.ndim < 2,
            ValueError,
            "Coefficients array must be at least 2-dimensional.",
        )
        errorif(x.ndim != 1, ValueError, "x must be 1-dimensional")
        errorif(x.size < 2, ValueError, "at least 2 breakpoints are needed")
        errorif(c.ndim < 2, ValueError, "c must have at least 2 dimensions")

        axis = axis % (c.ndim - 1)
        if extrapolate is None:
            extrapolate = True
        elif extrapolate != "periodic":
            extrapolate = bool(extrapolate)

        if axis != 0:
            # move the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis !=0 means that we have c.shape (..., k, m, ...)
            #                                                  ^
            #                                                 axis
            # So we roll two of them.
            c = jnp.moveaxis(c, axis + 1, 0)
            c = jnp.moveaxis(c, axis + 1, 0)

        errorif(
            c.shape[0] == 0,
            ValueError,
            "polynomial must be at least of order 0",
        )
        errorif(
            c.shape[1] != x.size - 1,
            ValueError,
            "number of coefficients != len(x)-1",
        )

        if check:

            dx = jnp.diff(x)
            errorif(
                jnp.any(dx < 0), ValueError, "`x` must be strictly increasing sequence."
            )

        self._extrapolate = extrapolate
        self._axis = axis
        self._x = x
        self._c = c

    @property
    def c(self) -> jax.Array:
        """Array of spline coefficients, shape(order, knots-1, ...)."""
        return self._c

    @property
    def x(self) -> jax.Array:
        """Array of knot values, shape(knots)."""
        return self._x

    @property
    def extrapolate(self) -> Union[bool, str]:
        """Whether to extrapolate beyond domain of known values."""
        return self._extrapolate

    @property
    def axis(self) -> int:
        """Axis along which to interpolate."""
        return self._axis

    @classmethod
    def construct_fast(
        cls,
        c: jax.Array,
        x: jax.Array,
        extrapolate: Union[bool, str] = None,
        axis: int = 0,
    ):
        """Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_extrapolate", extrapolate)
        object.__setattr__(self, "_axis", axis)
        return self

    @partial(jit, static_argnames=("nu", "extrapolate"))
    def __call__(self, x: jax.Array, nu: int = 0, extrapolate: Union[bool, str] = None):
        """Evaluate the piecewise polynomial or its derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = asarray_inexact(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = x.flatten()

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == "periodic":
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        # TODO: implement extrap

        i = jnp.clip(jnp.searchsorted(self.x, x, side="right"), 1, len(self.x) - 1)

        t = x - self.x[i - 1]
        c = self.c[:, i - 1]

        c = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(c.T).T
        y = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c.T, t).T

        y = y.reshape(x_shape + self.c.shape[2:])

        if not extrapolate:
            # x became 1d after flatten, so reshape this back to original x shape
            mask = jnp.logical_or(x > self.x[-1], x < self.x[0]).reshape(x_shape)
            y = jnp.where(mask.T, jnp.nan, y.T).T

        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(y.ndim))
            l = l[x_ndim : x_ndim + self.axis] + l[:x_ndim] + l[x_ndim + self.axis :]
            y = y.transpose(l)
        return y

    def derivative(self, nu: int = 1):
        """Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the derivative
            of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if nu < 0:
            return self.antiderivative(-nu)

        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(
                self.c.T
            ).T

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = jnp.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu: int = 1):
        """Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c.copy()
            for _ in range(nu):
                c2 = jnp.vectorize(jnp.polyint, signature="(n)->(m)")(c2.T).T
                # need to patch up continuity
                dx = jnp.diff(self.x)
                z = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c2.T, dx).T
                c2 = c2.at[-1, 1:].add(jnp.cumsum(z, axis=self.axis)[:-1])

        if self.extrapolate == "periodic":
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(c2, self.x, extrapolate, self.axis)

    def integrate(self, a: float, b: float, extrapolate: Union[bool, str] = None):
        """Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        integral = self.antiderivative(1)
        # Swap integration bounds if needed
        sign = 1 - 2 * (b < a)
        a, b = jnp.sort(jnp.array([a, b]))

        # Compute the integral.
        if extrapolate == "periodic":
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = jnp.divmod(interval, period)

            def truefun():
                return (integral(xe) - integral(xs)) * n_periods

            def falsefun():
                return (
                    jnp.zeros(self.c.shape[2:]) if self.c.shape[2:] else jnp.array(0.0)
                )

            out = jax.lax.cond(n_periods > 0, truefun, falsefun)

            # Map a to [xs, xe], b is always a + left.
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.

            def truefun(out):
                return out + (integral(b) - integral(a))

            def falsefun(out):
                out += integral(xe) - integral(a)
                out += integral(xs + left + a - xe) - integral(xs)
                return out

            out = jax.lax.cond(b <= xe, truefun, falsefun, out)
        else:
            out = integral(b, extrapolate=extrapolate) - integral(
                a, extrapolate=extrapolate
            )

        return sign * out.reshape(self.c.shape[2:])

    def solve(self, y=0.0, discontinuity=True, extrapolate=None):
        """Not currently implemented."""
        raise NotImplementedError

    def roots(self, discontinuity=True, extrapolate=None):
        """Not currently implemented."""
        raise NotImplementedError

    def extend(self, c, x, right=True):
        """Not currently implemented."""
        raise NotImplementedError

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """Not currently implemented."""
        raise NotImplementedError

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """Not currently implemented."""
        raise NotImplementedError


def prepare_input(x, y, axis, dydx=None, check=True):
    """Prepare input for cubic spline interpolators.

    All data are converted to numpy arrays and checked for correctness.
    Axes equal to `axis` of arrays `y` and `dydx` are moved to be the 0th
    axis. The value of `axis` is converted to lie in
    [0, number of dimensions of `y`).
    """
    x, y = map(asarray_inexact, (x, y))
    dx = jnp.diff(x)
    axis = axis % y.ndim
    errorif(
        jnp.issubdtype(x.dtype, jnp.complexfloating),
        ValueError,
        "`x` must contain real values.",
    )
    x = x.astype(float)

    if dydx is not None:
        dydx = asarray_inexact(dydx)
        errorif(
            y.shape != dydx.shape,
            ValueError,
            "The shapes of `y` and `dydx` must be identical.",
        )
        dtype = jnp.promote_types(y.dtype, dydx.dtype)
        dydx = dydx.astype(dtype)
        y = y.astype(dtype)
    if check:
        errorif(x.ndim != 1, ValueError, "`x` must be 1-dimensional.")
        errorif(x.shape[0] < 2, ValueError, "`x` must contain at least 2 elements.")
        errorif(
            x.shape[0] != y.shape[axis],
            ValueError,
            f"The length of `y` along `axis`={axis} doesn't match the length of `x`",
        )
        errorif(
            not jnp.all(jnp.isfinite(x)),
            ValueError,
            "`x` must contain only finite values.",
        )
        errorif(
            not jnp.all(jnp.isfinite(y)),
            ValueError,
            "`y` must contain only finite values.",
        )
        errorif(
            (dydx is not None) and (not jnp.all(jnp.isfinite(dydx))),
            ValueError,
            "`dydx` must contain only finite values.",
        )
        errorif(
            jnp.any(dx <= 0), ValueError, "`x` must be strictly increasing sequence."
        )

    return x, dx, y, axis, dydx


class CubicHermiteSpline(PPoly):
    """Piecewise-cubic interpolator matching values and first derivatives.

    The result is represented as a `PPoly` instance.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    dydx : array_like
        Array containing derivatives of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), it is set to True.
    check : bool
        Whether to perform checks on the input. Should be False if used under JIT.

    See Also
    --------
    Akima1DInterpolator : Akima 1D interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    """

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        dydx: jax.Array,
        axis: int = 0,
        extrapolate: Union[bool, str] = None,
        check: bool = True,
    ):
        if extrapolate is None:
            extrapolate = True

        x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx, check)

        y = jnp.moveaxis(y, axis, 0)
        dydx = jnp.moveaxis(dydx, axis, 0)
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        F = jnp.stack([y[:-1], y[1:], dydx[:-1] * dxr, dydx[1:] * dxr], axis=0).T
        c = jnp.vectorize(jnp.matmul, signature="(n,n),(n)->(n)")(A_CUBIC, F)[..., ::-1]
        # handle non-uniform spacing
        c = c / (dx[:, None] ** jnp.arange(4)[::-1])
        # c has shape (..., m, k) for m knots and order k
        c = c.T  # (k, m, ...)
        # c.shape = (k, m, ...), but we want it to be (..., k, m, ...)
        #                                                   ^
        #                                                  axis
        # So we roll two of them.
        c = jnp.moveaxis(c, 0, axis + 1)  # (m, ..., k)
        c = jnp.moveaxis(c, 0, axis + 1)  # (..., k, m, ...)
        super().__init__(c, x, extrapolate=extrapolate, axis=axis, check=check)


class PchipInterpolator(CubicHermiteSpline):
    r"""PCHIP 1-D monotonic cubic interpolation.

    ``x`` and ``y`` are arrays of values used to approximate some function f,
    with ``y = f(x)``. The interpolant uses monotonic cubic splines
    to find the value of new points. (PCHIP stands for Piecewise Cubic
    Hermite Interpolating Polynomial).

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        A 1-D array of monotonically increasing real values. ``x`` cannot
        include duplicate values (otherwise f is overspecified)
    y : ndarray, shape (..., npoints, ...)
        A N-D array of real values. ``y``'s length along the interpolation
        axis must be equal to the length of ``x``. Use the ``axis``
        parameter to select the interpolation axis.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    check : bool
        Whether to perform checks on the input. Should be False if used under JIT.

    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    Akima1DInterpolator : Akima 1D interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    The interpolator preserves monotonicity in the interpolation data and does
    not overshoot if the data is not smooth.

    The first derivatives are guaranteed to be continuous, but the second
    derivatives may jump at :math:`x_k`.

    Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
    by using PCHIP algorithm [1]_.

    Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
    are the slopes at internal points :math:`x_k`.
    If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
    them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
    weighted harmonic mean

    .. math::

        \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}

    where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.

    The end slopes are set using a one-sided scheme [2]_.

    References
    ----------
    .. [1] F. N. Fritsch and J. Butland,
           A method for constructing local
           monotone piecewise cubic interpolants,
           SIAM J. Sci. Comput., 5(2), 300-304 (1984).
           doi:`10.1137/0905021`.
    .. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.
           doi:`10.1137/1.9780898717952`

    """

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        axis: int = 0,
        extrapolate: Union[bool, str] = None,
        check: bool = True,
    ):
        x, _, y, axis, _ = prepare_input(x, y, axis, check=check)
        dydx = approx_df(x, y, "monotonic", axis=axis)
        super().__init__(x, y, dydx, axis=axis, extrapolate=extrapolate, check=check)


class Akima1DInterpolator(CubicHermiteSpline):
    """Akima interpolator.

    Fit piecewise cubic polynomials, given vectors x and y. The interpolation
    method by Akima uses a continuously differentiable sub-spline built from
    piecewise cubic polynomials. The resultant curve passes through the given
    data points and will appear smooth and natural.

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        1-D array of monotonically increasing real values.
    y : ndarray, shape (..., npoints, ...)
        N-D array of real values. The length of ``y`` along the interpolation axis
        must be equal to the length of ``x``. Use the ``axis`` parameter to
        select the interpolation axis.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    check : bool
        Whether to perform checks on the input. Should be False if used under JIT.

    See Also
    --------
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    Use only for precise data, as the fitted curve passes through the given
    points exactly. This routine is useful for plotting a pleasingly smooth
    curve through a few given points for purposes of plotting.

    References
    ----------
    [1] A new method of interpolation and smooth curve fitting based
        on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
        589-602.

    """

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        axis: int = 0,
        extrapolate: Union[bool, str] = None,
        check: bool = True,
    ):
        x, _, y, axis, _ = prepare_input(x, y, axis, check=check)
        t = approx_df(x, y, method="akima", axis=axis)
        super().__init__(x, y, t, axis=axis, extrapolate=extrapolate, check=check)


class CubicSpline(CubicHermiteSpline):
    """Cubic spline data interpolator.

    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.

        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:

        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.

        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple `(order, deriv_values)` allowing to specify arbitrary
        derivatives at curve ends:

        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array_like containing derivative values, shape must
          be the same as `y`, excluding ``axis`` dimension. For example, if
          `y` is 1-D, then `deriv_value` must be a scalar. If `y` is 3-D with
          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2-D
          and have the shape (n0, n1).
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.
    check : bool
        Whether to perform checks on the input. Should be False if used under JIT.

    See Also
    --------
    Akima1DInterpolator : Akima 1D interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    Parameters `bc_type` and ``extrapolate`` work independently, i.e. the
    former controls only construction of a spline, and the latter only
    evaluation.

    When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
    a condition that the first derivative is equal to the linear interpolant
    slope. When both boundary conditions are 'not-a-knot' and n = 3, the
    solution is sought as a parabola passing through given points.

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        axis: int = 0,
        bc_type: Union[str, tuple] = "not-a-knot",
        extrapolate: Union[bool, str] = None,
        check: bool = True,
    ):
        x, _, y, axis, _ = prepare_input(x, y, axis, check=check)
        df = approx_df(x, y, "cubic2", axis, bc_type=bc_type)
        super().__init__(x, y, df, axis=axis, extrapolate=extrapolate, check=check)
