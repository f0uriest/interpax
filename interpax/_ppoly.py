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

from collections.abc import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Inexact, Num, Real

from ._coefs import A_CUBIC
from ._fd_derivs import (
    _cubic2_system,
    _solve_cyclic_tridiagonal,
    _solve_tridiagonal,
    _validate_bc,
    approx_df,
)
from .utils import asarray_inexact, errorif, wrap_jit


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
        Polynomial breakpoints. Must be sorted in increasing order.
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

    _c: Inexact[Array, "k m ..."]
    _x: Float[Array, " m+1"]
    _extrapolate: bool | str = eqx.field(static=True)
    _axis: int = eqx.field(static=True)

    def __init__(
        self,
        c: Num[ArrayLike, "k m ..."],
        x: Real[ArrayLike, " m+1"],
        extrapolate: bool | str | None = None,
        axis: int = 0,
        check: bool = True,
    ) -> None:
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
    def c(self) -> Inexact[Array, "k m ..."]:
        """Array of spline coefficients, shape(order, knots-1, ...)."""
        return self._c

    @property
    def x(self) -> Float[Array, " m+1"]:
        """Array of knot values, shape(knots)."""
        return self._x

    @property
    def extrapolate(self) -> bool | str:
        """Whether to extrapolate beyond domain of known values."""
        return self._extrapolate

    @property
    def axis(self) -> int:
        """Axis along which to interpolate."""
        return self._axis

    @classmethod
    def construct_fast(
        cls,
        c: Inexact[ArrayLike, "k m ..."],
        x: Real[ArrayLike, " m+1"],
        extrapolate: bool | str | None = None,
        axis: int = 0,
    ) -> "PPoly":
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

    @wrap_jit(static_argnames=("nu", "extrapolate"))
    def __call__(
        self,
        x: Num[ArrayLike, "..."],
        nu: int = 0,
        extrapolate: bool | str | None = None,
    ) -> Inexact[Array, "..."]:
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

        i = jnp.clip(jnp.searchsorted(self.x, x, side="right"), 1, len(self.x) - 1)

        t = x - self.x[i - 1]
        c = self.c[:, i - 1]

        c = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(c.T).T
        y: jax.Array = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c.T, t).T

        y = y.reshape(x_shape + self.c.shape[2:])

        if not extrapolate:
            # x became 1d after flatten, so reshape this back to original x shape
            mask = jnp.logical_or(x > self.x[-1], x < self.x[0]).reshape(x_shape)
            y = jnp.where(mask.T, jnp.array(jnp.nan), y.T).T

        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(y.ndim))
            l = l[x_ndim : x_ndim + self.axis] + l[:x_ndim] + l[x_ndim + self.axis :]
            y = y.transpose(l)
        return y

    def derivative(self, nu: int = 1) -> "PPoly":
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

    def antiderivative(self, nu: int = 1) -> "PPoly":
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

    def integrate(
        self, a: Real, b: Real, extrapolate: bool | str | None = None
    ) -> Inexact[Array, ""]:
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

            def truefun1():
                return (integral(xe) - integral(xs)) * n_periods

            def falsefun1():
                return (
                    jnp.zeros(self.c.shape[2:]) if self.c.shape[2:] else jnp.array(0.0)
                )

            out = jax.lax.cond(n_periods > 0, truefun1, falsefun1)

            # Map a to [xs, xe], b is always a + left.
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.

            def truefun2(out):
                return out + (integral(b) - integral(a))

            def falsefun2(out):
                out += integral(xe) - integral(a)
                out += integral(xs + left + a - xe) - integral(xs)
                return out

            out = jax.lax.cond(b <= xe, truefun2, falsefun2, out)
        else:
            out = integral(b, extrapolate=extrapolate) - integral(
                a, extrapolate=extrapolate
            )

        return sign * out.reshape(self.c.shape[2:])

    @wrap_jit(static_argnames=("discontinuity", "extrapolate", "size"))
    def solve(
        self,
        y: Real[ArrayLike, "..."] = 0.0,
        discontinuity: bool = True,
        extrapolate: bool | str | None = None,
        size: int | None = None,
        fill_value: Real[ArrayLike, ""] = jnp.nan,
    ) -> Float[Array, "... size"]:
        """Find real solutions of the equation ``pp(x) == y``.

        Parameters
        ----------
        y : float or array_like, optional
            Right-hand side. Default is zero. If an array, it must be broadcastable
            to ``c.shape[2:]``.
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.
        size : int, optional
            Length of the returned array of roots. If fewer roots exist, the result
            is padded with ``fill_value``. If more exist, only the ``size`` smallest
            are returned. Default is the largest number of roots that can be
            reported for a piecewise polynomial of this order and number of
            intervals.
        fill_value : float, optional
            Value used to pad the result when fewer than ``size`` roots exist.
            Default is nan.

        Returns
        -------
        roots : ndarray, shape(..., size)
            Roots of the polynomial(s) in increasing order, followed by
            ``fill_value`` padding. Leading dimensions are ``c.shape[2:]``.

        Notes
        -----
        This routine works only on real-valued polynomials.

        If the piecewise polynomial contains sections that are
        identically equal to ``y``, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.

        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the ``discontinuity`` parameter is True.

        Roots are differentiable with respect to the coefficients, breakpoints
        and ``y``, except for roots reported at discontinuities or identically
        zero sections.
        """
        errorif(
            jnp.iscomplexobj(self.c),
            ValueError,
            "Root finding is only for real-valued polynomials",
        )
        if extrapolate is None:
            extrapolate = self.extrapolate
        extrapolate = False if extrapolate == "periodic" else bool(extrapolate)

        k, m = self.c.shape[:2]
        if size is None:
            # each interval gives at most k-1 roots or a (start, nan) pair for
            # identically zero sections, plus one root at a discontinuity.
            size = m * (max(k - 1, 2) + 1)

        c = self.c.reshape(k, m, -1)
        y = jnp.broadcast_to(asarray_inexact(y), self.c.shape[2:]).reshape(-1)
        fun = lambda c, y: _real_roots(
            c, self.x, y, discontinuity, extrapolate, size, fill_value
        )
        r = jax.vmap(fun, in_axes=(2, 0))(c, y)
        return r.reshape(self.c.shape[2:] + (size,))

    def roots(
        self,
        discontinuity: bool = True,
        extrapolate: bool | str | None = None,
        size: int | None = None,
        fill_value: Real[ArrayLike, ""] = jnp.nan,
    ) -> Float[Array, "... size"]:
        """Find real roots of the piecewise polynomial.

        Parameters
        ----------
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.
        size : int, optional
            Length of the returned array of roots. If fewer roots exist, the result
            is padded with ``fill_value``. If more exist, only the ``size`` smallest
            are returned. Default is the largest number of roots that can be
            reported for a piecewise polynomial of this order and number of
            intervals.
        fill_value : float, optional
            Value used to pad the result when fewer than ``size`` roots exist.
            Default is nan.

        Returns
        -------
        roots : ndarray, shape(..., size)
            Roots of the polynomial(s) in increasing order, followed by
            ``fill_value`` padding. Leading dimensions are ``c.shape[2:]``.

        See Also
        --------
        PPoly.solve
        """
        return self.solve(0.0, discontinuity, extrapolate, size, fill_value)

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


def _poly_real_roots(p: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Real roots of a polynomial, padded to a fixed length.

    Parameters
    ----------
    p : ndarray, shape(k,)
        Polynomial coefficients, highest order first. Must not be identically zero.

    Returns
    -------
    t : ndarray, shape(k-1,)
        Candidate roots.
    valid : ndarray of bool, shape(k-1,)
        Whether each entry of ``t`` is a real root.
    """
    d = p.shape[0] - 1
    idx = jnp.arange(d)
    nonzero = p != 0
    # Leading zero coefficients lower the degree. The monic polynomial of the true
    # degree goes in the top left block of a fixed size companion matrix, and the
    # remaining block is diagonal with eigenvalues outside the Cauchy bound on the
    # roots, so they can be discarded by magnitude.
    lead_idx = jnp.argmax(nonzero)
    deg = d - lead_idx
    lead = jnp.where(nonzero.any(), p[lead_idx], 1.0)
    b = jnp.take(p, lead_idx + 1 + idx, mode="fill", fill_value=0.0) / lead
    inblock = idx < deg
    bound = 1 + jnp.max(jnp.where(inblock, jnp.abs(b), 0.0), initial=0.0)
    C = jnp.zeros((d, d), dtype=b.dtype)
    C = C.at[0].set(jnp.where(inblock, -b, 0.0))
    C = C.at[idx[1:], idx[:-1]].set(jnp.where(inblock[1:], 1.0, 0.0))
    C = C + jnp.diag(jnp.where(inblock, 0.0, 2 * bound))
    sg = jax.lax.stop_gradient
    lam = jnp.linalg.eigvals(sg(C))
    eps = jnp.finfo(b.dtype).eps

    # Filters on the raw eigenvalues: discard the padding block, and keep one member
    # of each conjugate pair. Multiple roots are ill conditioned and come back as
    # clusters of complex eigenvalues with imaginary parts up to about eps**(1/j) for
    # multiplicity j, so the imaginary part is only a loose prefilter and the
    # residual below decides whether a candidate is a real root.
    candidate = (
        (jnp.abs(lam) <= 1.5 * bound)
        & (lam.imag >= 0)
        & (jnp.abs(lam.imag) <= eps**0.25 * sg(bound))
    )

    # One Newton step refines the root, kept only if it doesn't increase the
    # residual. This prevents a large step where p' is nearly zero from moving an
    # accurate eigenvalue away from the root.
    ps, dps = sg(p), sg(jnp.polyder(p))
    t0 = lam.real
    f0 = jnp.polyval(ps, t0)
    df0 = jnp.polyval(dps, t0)
    t1 = t0 - f0 / jnp.where(df0 == 0, 1.0, df0)
    f1 = jnp.polyval(ps, t1)
    accept = (df0 != 0) & (jnp.abs(f1) <= jnp.abs(f0))
    t = jnp.where(accept, t1, t0)
    f = jnp.where(accept, f1, f0)

    # A real root has a residual within the rounding error of evaluating p near it,
    # while the real part of a complex pair with imaginary part delta leaves a
    # residual of order p'' * delta**2.
    tol = 4 * p.shape[0] * eps * jnp.polyval(jnp.abs(ps), jnp.abs(t))
    valid = candidate & (jnp.abs(f) <= tol)

    # The eigenvalues carry no gradient, so the gradient comes from a Newton step
    # with zero value, which gives the implicit derivative -(dp/dtheta) / p'(t) at a
    # simple root.
    fg = jnp.polyval(p, t)
    dfg = jnp.polyval(jnp.polyder(p), t)
    step = jnp.where(dfg == 0, 0.0, fg / jnp.where(dfg == 0, 1.0, dfg))
    t = t - (step - sg(step))
    return t, valid


def _real_roots(
    c: jax.Array,
    x: jax.Array,
    y: jax.Array,
    discontinuity: bool,
    extrapolate: bool,
    size: int,
    fill_value: Real[ArrayLike, ""],
) -> jax.Array:
    """Sorted real roots of a single piecewise polynomial, padded to ``size``."""
    k, m = c.shape
    c = c.at[-1].add(-y)
    dx = jnp.diff(x)
    idx = jnp.arange(m)

    disc = jnp.zeros(m, dtype=bool)
    if discontinuity and m > 1:
        va = jax.vmap(jnp.polyval, in_axes=(1, 0))(c[:, :-1], dx[:-1])
        vb = c[-1, 1:]
        disc = disc.at[1:].set(((va < 0) & (vb > 0)) | ((va > 0) & (vb < 0)))
    zero = jnp.all(c == 0, axis=0)

    # Each interval fills slots [discontinuity, zero section start, nan marker,
    # roots...] in increasing order, so the flattened slots are sorted too.
    x0 = x[:-1, None]
    vals = [x0, x0, jnp.full_like(x0, jnp.nan)]
    valid = [disc[:, None], zero[:, None], zero[:, None]]
    if k > 1:
        t, tvalid = jax.vmap(_poly_real_roots, in_axes=1)(
            jnp.where(zero, 1.0, c)  # avoid nans from all zero coefficients
        )
        r = t + x0
        lo_ok = r >= x0
        hi_ok = r <= x[1:, None]
        if extrapolate:
            lo_ok = lo_ok | (idx == 0)[:, None]
            hi_ok = hi_ok | (idx == m - 1)[:, None]
        tvalid = tvalid & lo_ok & hi_ok & ~zero[:, None]
        key = jnp.where(tvalid, jax.lax.stop_gradient(r), jnp.inf)
        order = jnp.argsort(key, axis=1)
        vals.append(jnp.take_along_axis(r, order, axis=1))
        valid.append(jnp.take_along_axis(tvalid, order, axis=1))
    vals = jnp.concatenate(vals, axis=1).flatten()
    valid = jnp.concatenate(valid, axis=1).flatten()
    marker = jnp.zeros((m, vals.size // m), dtype=bool).at[:, 2].set(True).flatten()

    # Compact the valid entries, then drop repeats of the previous root, such as a
    # root on a breakpoint found in both neighboring intervals. A nan marker is kept
    # only if the zero section start before it is kept.
    order = jnp.argsort(~valid, stable=True)
    vals, valid, marker = vals[order], valid[order], marker[order]
    prev = jnp.concatenate([jnp.full_like(vals[:1], jnp.nan), vals[:-1]])
    keep = valid & (vals != prev)
    keep = jnp.where(marker, jnp.roll(keep, 1), keep)

    order = jnp.argsort(~keep, stable=True)
    vals, keep = vals[order], keep[order]
    if size > vals.size:
        pad = size - vals.size
        vals = jnp.concatenate([vals, jnp.full(pad, jnp.nan, dtype=vals.dtype)])
        keep = jnp.concatenate([keep, jnp.zeros(pad, dtype=bool)])
    return jnp.where(keep[:size], vals[:size], fill_value)


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
        x: Real[ArrayLike, " n"],
        y: Num[ArrayLike, " n ..."],
        dydx: Num[ArrayLike, " n ..."],
        axis: int = 0,
        extrapolate: bool | str | None = None,
        check: bool = True,
    ) -> None:
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
        x: Real[ArrayLike, " n"],
        y: Num[ArrayLike, " n ..."],
        axis: int = 0,
        extrapolate: bool | str | None = None,
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
        x: Real[ArrayLike, " n"],
        y: Num[ArrayLike, " n ..."],
        axis: int = 0,
        extrapolate: bool | str | None = None,
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
        x: Real[ArrayLike, " n"],
        y: Num[ArrayLike, " n ..."],
        axis: int = 0,
        bc_type: str | Iterable = "not-a-knot",
        extrapolate: bool | str | None = None,
        check: bool = True,
    ):
        x, _, y, axis, _ = prepare_input(x, y, axis, check=check)
        periodic = isinstance(bc_type, str) and bc_type == "periodic"
        if extrapolate is None:
            extrapolate = "periodic" if periodic else True
        if check and periodic:
            errorif(
                not jnp.allclose(
                    jnp.take(y, 0, axis), jnp.take(y, -1, axis), rtol=1e-15, atol=1e-15
                ),
                ValueError,
                f"The first and last `y` point along axis {axis} must be identical "
                "(within machine precision) when bc_type='periodic'.",
            )
        df = approx_df(x, y, "cubic2", axis, bc_type=bc_type)
        super().__init__(x, y, df, axis=axis, extrapolate=extrapolate, check=check)


# Banded matrices are stored cyclically: band[i, k + q] = A[i, (i + k) % n] for
# |k| <= q. Non-cyclic matrices keep the entries that wrap around equal to zero.


def _band_matmul(A, B):
    n = A.shape[0]
    qa, qb = (A.shape[1] - 1) // 2, (B.shape[1] - 1) // 2
    q = qa + qb
    out = jnp.zeros((n, 2 * q + 1), jnp.result_type(A, B))
    for a in range(-qa, qa + 1):
        Bs = jnp.roll(B, -a, axis=0)
        for b in range(-qb, qb + 1):
            out = out.at[:, a + b + q].add(A[:, a + qa] * Bs[:, b + qb])
    return out


def _band_transpose(A):
    q = (A.shape[1] - 1) // 2
    return jnp.stack([jnp.roll(A[:, q - k], -k) for k in range(-q, q + 1)], axis=1)


def _band_matvec(A, v):
    q = (A.shape[1] - 1) // 2
    shp = (-1,) + (1,) * (v.ndim - 1)
    out = jnp.zeros(v.shape, jnp.result_type(A, v))
    for k in range(-q, q + 1):
        out = out + A[:, k + q].reshape(shp) * jnp.roll(v, -k, axis=0)
    return out


def _band_pad(A, q):
    p = q - (A.shape[1] - 1) // 2
    return jnp.pad(A, ((0, 0), (p, p)))


def _band_to_dense(A):
    n, q = A.shape[0], (A.shape[1] - 1) // 2
    rows = np.arange(n)[:, None]
    cols = (rows + np.arange(-q, q + 1)) % n
    return jnp.zeros((n, n), A.dtype).at[rows, cols].add(A)


def _band_pin(A, J):
    # replace rows and columns J with those of the identity
    n, q = A.shape[0], (A.shape[1] - 1) // 2
    i = np.arange(n)[:, None]
    cols = (i + np.arange(-q, q + 1)) % n
    mask = np.isin(i, J) | np.isin(cols, J)
    return (A * ~mask).at[J, q].set(1)


def _band_columns(A, J):
    # columns J of a banded matrix, as a dense (n, len(J)) array
    n, q = A.shape[0], (A.shape[1] - 1) // 2
    out = jnp.zeros((n, len(J)), A.dtype)
    for c, j in enumerate(J):
        for k in range(-q, q + 1):
            i = (j - k) % n
            out = out.at[i, c].add(A[i, k + q])
    return out


def _solve_banded_symmetric(A, b):
    # Block LU of a non-cyclic symmetric banded matrix, treated as block tridiagonal
    # with blocks of size q and pivoting only within blocks. Unlike Cholesky this
    # needs no square roots, so roundoff in severely ill conditioned systems degrades
    # the solution gradually instead of producing NaN.
    n, w = A.shape
    q = (w - 1) // 2
    nb = -(-n // q)
    pad = nb * q - n
    A = jnp.concatenate([A, jnp.zeros((pad, w), A.dtype).at[:, q].set(1)])
    b = jnp.concatenate([b, jnp.zeros((pad,) + b.shape[1:], b.dtype)])
    r = np.arange(q)
    rows = np.arange(nb)[:, None, None] * q + r[None, :, None]
    off = r[None, None, :] - r[None, :, None]
    D = A[rows, off + q]  # diagonal blocks
    Lo = jnp.where(off >= 0, A[rows[1:], np.maximum(off, 0)], 0)  # subdiagonal
    dtype = jnp.result_type(A, b)
    D = D.astype(dtype)
    Lo = Lo.astype(dtype)
    rb = b.reshape(nb, q, -1).astype(dtype)

    def forward(carry, xs):
        Sp, zp = carry
        Di, Li, ri = xs
        G = jnp.linalg.solve(Sp, jnp.concatenate([Li.T, zp], axis=1))
        S = Di - Li @ G[:, :q]
        z = ri - Li @ G[:, q:]
        return (S, z), (S, z)

    _, (Ss, zs) = jax.lax.scan(forward, (D[0], rb[0]), (D[1:], Lo, rb[1:]))
    Ss = jnp.concatenate([D[:1], Ss])
    zs = jnp.concatenate([rb[:1], zs])

    def backward(xn, xs):
        S, Ln, z = xs
        xi = jnp.linalg.solve(S, z - Ln.T @ xn)
        return xi, xi

    xl = jnp.linalg.solve(Ss[-1], zs[-1])
    _, xs = jax.lax.scan(backward, xl, (Ss[:-1], Lo, zs[:-1]), reverse=True)
    x = jnp.concatenate([xs, xl[None]]).reshape((nb * q,) + b.shape[1:])
    return x[:n]


def _solve_cyclic_banded_spd(A, b):
    # Solve K x = b for symmetric positive definite K in cyclic banded storage.
    n, q = A.shape[0], (A.shape[1] - 1) // 2
    if n <= 4 * q:
        return jnp.linalg.solve(_band_to_dense(A), b.astype(jnp.result_type(A, b)))
    # The entries that wrap around form a corner block C = K[:q, -q:]. Writing
    # K = Kt - Y G Y^T with Y = [I; -C^T / g] and G = g I gives a non-cyclic banded
    # Kt = K + Y G Y^T, which is positive definite since K is, so x follows from
    # banded solves and the Woodbury identity. Taking g = max|C| keeps the update
    # comparable to the entries it replaces, so the Woodbury correction doesn't
    # suffer from cancellation.
    k = np.arange(-q, q + 1)
    i = np.arange(n)[:, None]
    wrap = (i + k < 0) | (i + k >= n)
    a, kk = np.nonzero(wrap[:q])
    C = jnp.zeros((q, q), A.dtype).at[a, a + k[kk] + q].add(A[a, kk])
    g = jnp.max(jnp.abs(C))
    # a corner that is negligible relative to K is dropped instead
    coupled = g > jnp.finfo(A.dtype).eps * jnp.max(jnp.abs(A))
    g = jnp.where(coupled, g, 1)
    At = (A * ~wrap).at[:q, q].add(coupled * g)
    a, c = np.nonzero(np.ones((q, q)))
    At = At.at[n - q + a, c - a + q].add(coupled * (C.T @ C)[a, c] / g)
    Y = jnp.zeros((n, q), A.dtype).at[:q].set(jnp.eye(q)).at[n - q :].set(-C.T / g)
    Z = _solve_banded_symmetric(At, Y)
    z = _solve_banded_symmetric(At, b)
    cap = jnp.where(coupled, jnp.eye(q) / g - Y.T @ Z, jnp.eye(q))
    corr = Z @ jnp.linalg.solve(cap, (Y.T @ z.reshape(n, -1)).astype(z.dtype))
    return z + coupled * corr.reshape(z.shape)


def _smoothing_null_space(Bt, bc, n, periodic):
    # Basis Z for the null space of Q, given its interval form Bt, and indices J
    # with Z_J invertible.
    N = Bt.shape[0]
    # Q v = 0 means Bt^T v = 0, which makes v constant over the interior rows, so the
    # null space Z of Q is set by the end rows. It contains a vector concentrated at
    # an end exactly when straight lines satisfy the homogeneous boundary condition
    # there, and otherwise a single vector spanning both ends. J holds one index per
    # null vector where Z_J is invertible.
    if periodic:
        Z = jnp.ones((N, 1), Bt.dtype)
        J = np.array([0])
    else:
        line_ok = [isinstance(b, str) or b[0] == 2 for b in bc]
        if n == 3 and bc[0] == "not-a-knot" and bc[1] == "not-a-knot":
            line_ok = [True, True]
        if all(line_ok):
            J = np.array([0, N - 1])
        else:
            J = np.array([N - 1 if line_ok[1] else 0])
        if n >= 5:
            # the end rows only reach intervals 0, 1 and n-3, n-2, giving
            #   v[1] = a0 v[0],  v[i] = c for 2 <= i <= n-3,  c = (a0 + a1) v[0]
            #   v[n-2] = -b1 v[n-1],  c = -(b0 + b1) v[n-1]
            # where a and b are the interval weights of the first and last rows
            a0, a1 = Bt[0, 2], Bt[0, 3]
            b0, b1 = Bt[-1, 0], Bt[-1, 1]
            a = 0 if line_ok[0] else a0 + a1
            b = 0 if line_ok[1] else -(b0 + b1)

            def null_vector(v0, c, vn):
                z = jnp.full(N, c, Bt.dtype)
                z = z.at[0].set(v0).at[1].set(a0 * v0)
                return z.at[-2].set(-b1 * vn).at[-1].set(vn)

            if all(line_ok):
                Z = jnp.stack([null_vector(1, 0, 0), null_vector(0, 0, 1)], axis=1)
            elif line_ok[1]:
                Z = null_vector(0, 0, 1)[:, None]
            elif line_ok[0]:
                Z = null_vector(1, 0, 0)[:, None]
            else:
                Z = null_vector(1 / a, 1, 1 / b)[:, None]
        else:
            Btd = _band_to_dense(Bt)[:, : n - 1]
            Z = jnp.linalg.svd(Btd.T)[2][-len(J) :].T

    return Z, J


@eqx.filter_jit
def _smoothing_spline(x, y, w, p, bc):
    # Knot values and slopes of the smoothing spline, with y varying along axis 0
    # and bc as returned by _validate_bc.
    dtype = y.dtype
    n = x.size
    h = jnp.diff(x)
    periodic = isinstance(bc[0], str) and bc[0] == "periodic" and n > 2
    rdtype = h.dtype

    # For periodic splines the last knot duplicates the first, leaving N = n-1
    # independent knots, each data site contributing to the knot it falls on.
    N = n - 1 if periodic else n
    knot = jnp.arange(n) % N
    wN = jnp.zeros(N, w.dtype).at[knot].add(w)
    shp = (-1,) + (1,) * (y.ndim - 1)
    yN = jnp.zeros((N,) + y.shape[1:], dtype).at[knot].add(w.reshape(shp) * y)
    yN = (yN / wN.reshape(shp)).reshape(N, -1)

    # The spline is parametrized by knot values f and second derivatives m, so
    # int f''^2 = m^T M m with M tridiagonal, and C1 continuity at interior knots
    # reads M m = Q^T f. Matrices are banded, with cyclic wrap when periodic.
    iv = jnp.arange(n - 1) % N
    jv = (iv + 1) % N
    M = jnp.zeros((N, 3), rdtype)
    M = M.at[iv, 1].add(h / 3).at[jv, 1].add(h / 3)
    M = M.at[iv, 2].add(h / 6).at[jv, 0].add(h / 6)
    Qt = jnp.zeros((N, 5), rdtype)
    Qt = Qt.at[iv, 3].add(1 / h).at[iv, 2].add(-1 / h)
    Qt = Qt.at[jv, 2].add(-1 / h).at[jv, 1].add(1 / h)
    # The same conditions in terms of the interval slopes D = (f[j] - f[i]) / h,
    # Q^T f = Bt D, where Bt is banded over intervals. Applying Q this way takes
    # differences before dividing by h, which avoids cancellation when knots are
    # closely spaced.
    Bt = jnp.zeros((N, 5), rdtype).at[iv, 2].add(1).at[jv, 1].add(-1)
    A = M
    g = jnp.zeros((N, yN.shape[1]), dtype)

    if not periodic:
        # The boundary conditions replace the C1 condition in the first and last
        # rows. They come from the C2 system for the knot slopes s, whose end
        # rows involve two slopes each, rewritten using
        #   s[i] = D - h (2 m[i] + m[i+1]) / 6
        #   s[i+1] = D + h (m[i] + 2 m[i+1]) / 6
        # on the end intervals, where D = (f[i+1] - f[i]) / h.
        lower, diag, upper, _, _ = _cubic2_system(x, y, bc)
        ends = np.unique(np.r_[np.arange(min(3, n)), np.arange(max(n - 3, 0), n)])

        def b_ends(v):
            f = jnp.zeros(y.shape, dtype).at[ends].set(v.reshape(shp).astype(dtype))
            b = _cubic2_system(x, f, bc)[3]
            return b[np.array([0, -1])].reshape(2, -1)[:, 0]

        # each end row of the slope system depends on at most 3 knot values, so
        # its dependence on f is found exactly by probing only the end knots
        G = jax.jacfwd(b_ends)(jnp.zeros(len(ends), rdtype)).real
        b0 = _cubic2_system(x, jnp.zeros_like(y), bc)[3]
        b0 = b0[np.array([0, -1])].reshape(2, -1)
        for e, (row, lo, c0, c1, he) in enumerate(
            [
                (0, 0, diag[0], upper[0], h[0]),
                (n - 1, n - 2, lower[-1], diag[-1], h[-1]),
            ]
        ):
            am = jnp.stack([he * (c1 - 2 * c0), he * (2 * c1 - c0)]) / 6
            qrow = jnp.zeros(5, rdtype)
            for col, val in zip(ends, G[e]):
                if abs(col - row) <= 2:
                    qrow = qrow.at[col - row + 2].add(val)
            qrow = qrow.at[lo - row + 2].add((c0 + c1) / he)
            qrow = qrow.at[lo + 1 - row + 2].add(-(c0 + c1) / he)
            # cancellation between the two terms leaves roundoff where the
            # condition doesn't involve f at all, which is removed here
            tol = 16 * jnp.finfo(rdtype).eps * jnp.abs(c0 + c1) / he
            qrow = jnp.where(jnp.abs(qrow) <= tol, 0, qrow)
            # rescale the row to match the others, which doesn't change the
            # condition but keeps the linear system well balanced
            scale = M[row, 1] / jnp.max(jnp.abs(am))
            A = A.at[row].set(0).at[row, lo - row + 1].set(scale * am[0])
            A = A.at[row, lo + 1 - row + 1].set(scale * am[1])
            Qt = Qt.at[row].set(scale * qrow)
            # the row annihilates constants, so it's a combination of interval
            # slopes with weights h[i] * sum(qrow[c] for c > i)
            brow = jnp.zeros(5, rdtype)
            last = min(row + 2, n - 1)
            for i in range(max(row - 2, 0), last):
                tail = sum(qrow[c - row + 2] for c in range(i + 1, last + 1))
                brow = brow.at[i - row + 2].set(h[i] * tail)
            Bt = Bt.at[row].set(scale * brow)
            g = g.at[row].set(scale * b0[e])

    # A differs from M only in the end rows, so with A = M + U D^T where U picks
    # the first and last rows, A M^-1 A^T = A + A^T - M + U (D^T M^-1 D) U^T,
    # which is banded apart from the coupling between the first and last rows.
    AMA = A + _band_transpose(A) - M
    if not periodic:
        # columns of D, i.e. the first and last rows of A - M
        k = np.arange(-1, 2)
        Dm = jnp.zeros((N, 2), rdtype)
        Dm = Dm.at[k % N, 0].add((A - M)[0]).at[(N - 1 + k) % N, 1].add((A - M)[-1])
        S = Dm.T @ _solve_tridiagonal(M[1:, 0], M[:, 1], M[:-1, 2], Dm)
        S01 = (S[0, 1] + S[1, 0]) / 2
        AMA = AMA.at[0, 1].add(S[0, 0]).at[-1, 1].add(S[1, 1])
        AMA = AMA.at[0, 0].add(S01).at[-1, 2].add(S01)

    # normalize smoothing parameter so it is insensitive to the scale and
    # number of data sites
    span = jnp.ptp(x)
    eff_x = 1 + (span**2) / jnp.sum(h**2)
    eff_w = jnp.sum(w) ** 2 / jnp.sum(w**2)
    k = 80 * (span**3) * (n**-2) * (eff_x**-0.5) * (eff_w**-0.5)
    sp = p
    pk = sp / (sp + (1 - sp) * k)

    # Minimizing pk (f-y)^T W (f-y) + (1-pk) m^T M m subject to A m = Q^T f + g
    # gives f = y - (1-pk) W^-1 Q u and m = pk M^-1 A^T u, where
    #   K u = (pk P + (1-pk) R) u = Q^T y + g,  P = A M^-1 A^T,  R = Q^T W^-1 Q.
    # P is positive definite but R is singular, so as pk -> 0, u diverges along the
    # null space of Q while f and m have finite limits.
    Qtw = Qt * jnp.stack([jnp.roll(1 / wN, -kk) for kk in range(-2, 3)], axis=1)
    P = _band_pad(AMA, 4)
    R = _band_matmul(Qtw, _band_transpose(Qt))
    K = pk * P + (1 - pk) * R
    hr = h[:, None]
    Dy = jnp.zeros_like(yN).at[iv].set((yN[jv] - yN[iv]) / hr)
    r = _band_matvec(Bt, Dy) + g

    def apply_Q(v):
        # Q v, taking differences before dividing by h
        t = _band_matvec(_band_transpose(Bt), v)[iv] / hr
        return jnp.zeros_like(v).at[jv].add(t).at[iv].add(-t)

    Z, J = _smoothing_null_space(Bt, bc, n, periodic)

    def solve_free(B, rhs):
        # solve with the rows and columns J removed, returning zeros in J
        return _solve_cyclic_banded_spd(_band_pin(B, J), rhs.at[J].set(0))

    # With u = Z c + v, v_J = 0 and R Z = 0, the system becomes K v + P Z ct = r for
    # ct = pk c, which stays finite as pk -> 0. Eliminating v leaves T ct = rho.
    PZ = _band_matvec(P, Z)
    KJ = _band_columns(K, J)
    vr = solve_free(K, r)
    vp = solve_free(K, PZ)
    T = PZ[J] - KJ.T @ vp
    rho = r[J] - KJ.T @ vr
    ct = jnp.linalg.solve(T.astype(jnp.result_type(T, rho)), rho)
    v = vr - vp @ ct
    f = yN - (1 - pk) * apply_Q(v) / wN[:, None]
    Atu = _band_matvec(_band_transpose(A), Z @ ct + pk * v)
    if periodic:
        m = _solve_cyclic_tridiagonal(M[:, 0], M[:, 1], M[:, 2], Atu)
    else:
        m = _solve_tridiagonal(M[1:, 0], M[:, 1], M[:-1, 2], Atu)

    f, m = f[knot], m[knot]
    D = (f[1:] - f[:-1]) / hr
    s = jnp.concatenate(
        [
            D - hr * (2 * m[:-1] + m[1:]) / 6,
            D[-1:] + hr[-1:] * (m[-2:-1] + 2 * m[-1:]) / 6,
        ]
    )
    return f.reshape(y.shape), s.reshape(y.shape)


class CubicSmoothingSpline(CubicSpline):
    """Smoothing spline for noisy data.

    The spline f minimizes::

        p̂ ∑ᵢ wᵢ ||yᵢ − f(xᵢ)||²  + (1−p̂) ∫ₓ ||f''(x)||²

    where p̂ is the smoothing parameter ``p`` rescaled according to the spacing of
    ``x`` and the weights ``w``, so that a given ``p`` produces a similar amount of
    smoothing regardless of the scale and number of data sites. The normalization
    of ``p`` follows the csaps package [1]_, and the spline is computed with a
    generalization of the method of Reinsch [2]_, [3]_ to other boundary conditions.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like, shape(n,...)
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    p : float, optional
        Normalized smoothing parameter in the range [0,1]. Default is 0.5.

        - For ``p=0`` the spline is the limit of small ``p``: the spline with the least
          integrated squared second derivative that satisfies the boundary
          conditions, with any remaining freedom fit to the data by weighted least
          squares. For natural or not-a-knot boundary conditions this is the weighted
          least squares straight line.
        - For ``p=1``, it is the cubic spline interpolant.
    w : array_like, shape(n,)
        Weights for spline fitting. Must be positive. If None, then weights are all
        equal. Default is None.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment.

        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:

        * 'not-a-knot': The first and second segment at a curve end
          are the same polynomial.
        * 'periodic': The spline is periodic with period ``x[-1] - x[0]``, so
          its value and first two derivatives match at both ends. ``y[0]`` and
          ``y[-1]`` are both fit by the value at ``x[0]`` and need not be equal.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural' (default): The second derivative at curve ends are zero. Assuming
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

    References
    ----------
    .. [1] E. Prilepin, `csaps: Cubic spline approximation (smoothing)
            <https://github.com/espdev/csaps>`_.
    .. [2] C. H. Reinsch, "Smoothing by spline functions", Numerische Mathematik,
            10, pp. 177-183, 1967.
    .. [3] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """

    def __init__(
        self,
        x: Real[ArrayLike, " n"],
        y: Num[ArrayLike, " n ..."],
        p: float | None = None,
        w: Real[ArrayLike, " n"] | None = None,
        axis: int = 0,
        bc_type: str | Iterable = "natural",
        extrapolate: bool | str | None = None,
        check: bool = True,
    ):
        if check and p is not None:
            errorif(not 0 <= p <= 1, ValueError, f"p must be in [0, 1], got {p}.")
        x = asarray_inexact(x)
        y = jnp.moveaxis(asarray_inexact(y), axis, 0)
        w = jnp.ones_like(x) if w is None else asarray_inexact(w)
        bc, dtype = _validate_bc(bc_type, y.shape[1:], y.dtype)
        y = y.astype(dtype)
        periodic = isinstance(bc[0], str) and bc[0] == "periodic" and x.size > 2
        f, s = _smoothing_spline(x, y, w, jnp.asarray(0.5 if p is None else p), bc)
        if extrapolate is None:
            extrapolate = "periodic" if periodic else True
        CubicHermiteSpline.__init__(
            self,
            x,
            jnp.moveaxis(f, 0, axis),
            jnp.moveaxis(s, 0, axis),
            axis=axis,
            extrapolate=extrapolate,
            check=check,
        )
