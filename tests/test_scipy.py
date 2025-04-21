"""Tests for scipy API.

Tests mostly copied from scipy with minor rewrites.

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

import io
import warnings

import numpy as np
import pytest
import scipy.interpolate
from jax import config as jax_config
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)
from pytest import raises as assert_raises
from scipy.interpolate import splev, splint, splrep

from interpax import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
    PPoly,
)

jax_config.update("jax_enable_x64", True)


class TestAkima1DInterpolator:
    def test_eval(self):
        x = np.arange(0.0, 11.0)
        y = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 6.0, 5.5, 5.5, 2.7, 5.1, 3.0])
        ak = Akima1DInterpolator(x, y)
        xi = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2, 8.6, 9.9, 10.0]
        )
        yi = np.array(
            [
                0.0,
                1.375,
                2.0,
                1.5,
                1.953125,
                2.484375,
                4.1363636363636366866103344,
                5.9803623910336236590978842,
                5.5067291516462386624652936,
                5.2031367459745245795943447,
                4.1796554159017080820603951,
                3.4110386597938129327189927,
                3.0,
            ]
        )
        assert_allclose(ak(xi), yi)

    def test_eval_2d(self):
        x = np.arange(0.0, 11.0)
        y = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 6.0, 5.5, 5.5, 2.7, 5.1, 3.0])
        y = np.column_stack((y, 2.0 * y))
        ak = Akima1DInterpolator(x, y)
        xi = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2, 8.6, 9.9, 10.0]
        )
        yi = np.array(
            [
                0.0,
                1.375,
                2.0,
                1.5,
                1.953125,
                2.484375,
                4.1363636363636366866103344,
                5.9803623910336236590978842,
                5.5067291516462386624652936,
                5.2031367459745245795943447,
                4.1796554159017080820603951,
                3.4110386597938129327189927,
                3.0,
            ]
        )
        yi = np.column_stack((yi, 2.0 * yi))
        assert_allclose(ak(xi), yi)

    def test_eval_3d(self):
        x = np.arange(0.0, 11.0)
        y_ = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 6.0, 5.5, 5.5, 2.7, 5.1, 3.0])
        y = np.empty((11, 2, 2))
        y[:, 0, 0] = y_
        y[:, 1, 0] = 2.0 * y_
        y[:, 0, 1] = 3.0 * y_
        y[:, 1, 1] = 4.0 * y_
        ak = Akima1DInterpolator(x, y)
        xi = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2, 8.6, 9.9, 10.0]
        )
        yi = np.empty((13, 2, 2))
        yi_ = np.array(
            [
                0.0,
                1.375,
                2.0,
                1.5,
                1.953125,
                2.484375,
                4.1363636363636366866103344,
                5.9803623910336236590978842,
                5.5067291516462386624652936,
                5.2031367459745245795943447,
                4.1796554159017080820603951,
                3.4110386597938129327189927,
                3.0,
            ]
        )
        yi[:, 0, 0] = yi_
        yi[:, 1, 0] = 2.0 * yi_
        yi[:, 0, 1] = 3.0 * yi_
        yi[:, 1, 1] = 4.0 * yi_
        assert_allclose(ak(xi), yi)

    def test_degenerate_case_multidimensional(self):
        # This test is for issue #5683.
        x = np.array([0, 1, 2])
        y = np.vstack((x, x**2)).T
        ak = Akima1DInterpolator(x, y)
        x_eval = np.array([0.5, 1.5])
        y_eval = ak(x_eval)
        assert_allclose(y_eval, np.vstack((x_eval, x_eval**2)).T)

    def test_extend(self):
        x = np.arange(0.0, 11.0)
        y = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 6.0, 5.5, 5.5, 2.7, 5.1, 3.0])
        ak = Akima1DInterpolator(x, y)
        with pytest.raises(NotImplementedError):
            ak.extend(None, None)


class TestPPolyCommon:
    # test basic functionality for PPoly and BPoly
    def test_sort_check(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 1, 0.5])
        assert_raises(ValueError, PPoly, c, x)

    def test_ctor_c(self):
        # wrong shape: `c` must be at least 2D
        with assert_raises(ValueError):
            PPoly([1, 2], [0, 1])

    def test_extend(self):
        # Test adding new points to the piecewise polynomial
        np.random.seed(1234)

        order = 3
        x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])
        c = 2 * np.random.rand(order + 1, len(x) - 1, 2, 3) - 1

        for cls in (PPoly,):
            pp = cls(c[:, :9], x[:10])
            with pytest.raises(NotImplementedError):
                pp.extend(None, None)

    def test_shape(self):
        np.random.seed(1234)
        c = np.random.rand(8, 12, 5, 6, 7)
        x = np.sort(np.random.rand(13))
        xp = np.random.rand(3, 4)
        for cls in (PPoly,):
            p = cls(c, x)
            assert_equal(p(xp).shape, (3, 4, 5, 6, 7))

        # 'scalars'
        for cls in (PPoly,):
            p = cls(c[..., 0, 0, 0], x)

            assert_equal(np.shape(p(0.5)), ())
            assert_equal(np.shape(p(np.array(0.5))), ())

            assert_raises(TypeError, p, np.array([[0.1, 0.2], [0.4]], dtype=object))

    def test_complex_coef(self):
        np.random.seed(12345)
        x = np.sort(np.random.random(13))
        c = np.random.random((8, 12)) * (1.0 + 0.3j)
        c_re, c_im = c.real, c.imag
        xp = np.random.random(5)
        for cls in (PPoly,):
            p, p_re, p_im = cls(c, x), cls(c_re, x), cls(c_im, x)
            for nu in [0, 1, 2]:
                assert_allclose(p(xp, nu).real, p_re(xp, nu))
                assert_allclose(p(xp, nu).imag, p_im(xp, nu))

    def test_axis(self):
        np.random.seed(12345)
        c = np.random.rand(3, 4, 5, 6, 7, 8)
        c_s = c.shape
        xp = np.random.random((1, 2))
        for axis in (0, 1, 2, 3):
            m = c.shape[axis + 1]
            x = np.sort(np.random.rand(m + 1))
            for cls in (PPoly,):
                p = cls(c, x, axis=axis)
                assert_equal(
                    p.c.shape, c_s[axis : axis + 2] + c_s[:axis] + c_s[axis + 2 :]
                )
                res = p(xp)
                targ_shape = c_s[:axis] + xp.shape + c_s[2 + axis :]
                assert_equal(res.shape, targ_shape)

                # deriv/antideriv does not drop the axis
                for p1 in [
                    cls(c, x, axis=axis).derivative(),
                    cls(c, x, axis=axis).derivative(2),
                    cls(c, x, axis=axis).antiderivative(),
                    cls(c, x, axis=axis).antiderivative(2),
                ]:
                    assert_equal(p1.axis, p.axis)

        # c array needs two axes for the coefficients and intervals, so
        # we expect 0 <= axis < c.ndim-1; raise otherwise
        for axis in (-1, 4, 5, 6):
            for cls in (PPoly,):
                assert_raises(ValueError, cls, **dict(c=c, x=x, axis=axis))


class TestPolySubclassing:
    class P(PPoly):
        pass

    class B(P):
        pass

    def _make_polynomials(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(3))
        c = np.random.random((4, 2))
        return self.P(c, x), self.B(c, x)

    def test_derivative(self):
        pp, bp = self._make_polynomials()
        for p in (pp, bp):
            pd = p.derivative()
            assert_equal(p.__class__, pd.__class__)

        ppa = pp.antiderivative()
        assert_equal(pp.__class__, ppa.__class__)


class TestPPoly:
    def test_simple(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x)

        assert_allclose(p(0.3), 1 * 0.3**2 + 2 * 0.3 + 3)
        assert_allclose(p(0.7), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)

    def test_periodic(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x, extrapolate="periodic")

        assert_allclose(p(1.3), 1 * 0.3**2 + 2 * 0.3 + 3)
        assert_allclose(p(-0.3), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)

        assert_allclose(p(1.3, 1), 2 * 0.3 + 2)
        assert_allclose(p(-0.3, 1), 8 * (0.7 - 0.5) + 5)

    def test_read_only(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        xnew = np.array([0, 0.1, 0.2])
        PPoly(c, x, extrapolate="periodic")

        for writeable in (True, False):
            x.flags.writeable = writeable
            c.flags.writeable = writeable
            f = PPoly(c, x)
            vals = f(xnew)
            assert_(np.isfinite(vals).all())

    def test_multi_shape(self):
        c = np.random.rand(6, 2, 1, 2, 3)
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x)
        assert_equal(p.x.shape, x.shape)
        assert_equal(p.c.shape, c.shape)
        assert_equal(p(0.3).shape, c.shape[2:])

        assert_equal(p(np.random.rand(5, 6)).shape, (5, 6) + c.shape[2:])

        dp = p.derivative()
        assert_equal(dp.c.shape, (5, 2, 1, 2, 3))
        ip = p.antiderivative()
        assert_equal(ip.c.shape, (7, 2, 1, 2, 3))

    def test_construct_fast(self):
        np.random.seed(1234)
        c = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
        x = np.array([0, 0.5, 1])
        p = PPoly.construct_fast(c, x)
        assert_allclose(p(0.3), 1 * 0.3**2 + 2 * 0.3 + 3)
        assert_allclose(p(0.7), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)

    def test_vs_alternative_implementations(self):
        np.random.seed(1234)
        c = np.random.rand(3, 12, 22)
        x = np.sort(np.r_[0, np.random.rand(11), 1])

        p = PPoly(c, x)

        xp = np.r_[0.3, 0.5, 0.33, 0.6]
        expected = _ppoly_eval_1(c, x, xp)
        assert_allclose(p(xp), expected)

        expected = _ppoly_eval_2(c[:, :, 0], x, xp)
        assert_allclose(p(xp)[:, 0], expected)

    def test_derivative_simple(self):
        np.random.seed(1234)
        c = np.array([[4, 3, 2, 1]]).T
        dc = np.array([[3 * 4, 2 * 3, 2]]).T
        ddc = np.array([[2 * 3 * 4, 1 * 2 * 3]]).T
        x = np.array([0, 1])

        pp = PPoly(c, x)
        dpp = PPoly(dc, x)
        ddpp = PPoly(ddc, x)

        assert_allclose(pp.derivative().c, dpp.c)
        assert_allclose(pp.derivative(2).c, ddpp.c)

    def test_derivative_eval(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0)
        spp = scipy.interpolate.PPoly.from_spline(spl)
        pp = PPoly(spp.c, spp.x)

        xi = np.linspace(0, 1, 200)
        for dx in range(0, 3):
            assert_allclose(pp(xi, dx), splev(xi, spl, dx))

    def test_derivative(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0, k=5)
        spp = scipy.interpolate.PPoly.from_spline(spl)
        pp = PPoly(spp.c, spp.x)

        xi = np.linspace(0, 1, 200)
        for dx in range(0, 10):
            assert_allclose(pp(xi, dx), pp.derivative(dx)(xi), err_msg="dx=%d" % (dx,))

    def test_antiderivative_of_constant(self):
        # https://github.com/scipy/scipy/issues/4216
        p = PPoly([[1.0]], [0, 1])
        assert_array_equal(p.antiderivative().c, PPoly([[1], [0]], [0, 1]).c)
        assert_array_equal(p.antiderivative().x, PPoly([[1], [0]], [0, 1]).x)

    def test_antiderivative_regression_4355(self):
        # https://github.com/scipy/scipy/issues/4355
        p = PPoly([[1.0, 0.5]], [0, 1, 2])
        q = p.antiderivative()
        assert_array_equal(q.c, [[1, 0.5], [0, 1]])
        assert_array_equal(q.x, [0, 1, 2])
        assert_allclose(p.integrate(0, 2), 1.5)
        assert_allclose(q(2) - q(0), 1.5)

    def test_antiderivative_simple(self):
        np.random.seed(1234)
        # [ p1(x) = 3*x**2 + 2*x + 1,
        #   p2(x) = 1.6875]
        c = np.array([[3, 2, 1], [0, 0, 1.6875]]).T
        # [ pp1(x) = x**3 + x**2 + x,
        #   pp2(x) = 1.6875*(x - 0.25) + pp1(0.25)]
        ic = np.array([[1, 1, 1, 0], [0, 0, 1.6875, 0.328125]]).T
        # [ ppp1(x) = (1/4)*x**4 + (1/3)*x**3 + (1/2)*x**2,
        #   ppp2(x) = (1.6875/2)*(x - 0.25)**2 + pp1(0.25)*x + ppp1(0.25)]
        iic = np.array(
            [
                [1 / 4, 1 / 3, 1 / 2, 0, 0],
                [0, 0, 1.6875 / 2, 0.328125, 0.037434895833333336],
            ]
        ).T
        x = np.array([0, 0.25, 1])

        pp = PPoly(c, x)
        ipp = pp.antiderivative()
        iipp = pp.antiderivative(2)
        iipp2 = ipp.antiderivative()

        assert_allclose(ipp.x, x)
        assert_allclose(ipp.c.T, ic.T)
        assert_allclose(iipp.c.T, iic.T)
        assert_allclose(iipp2.c.T, iic.T)

    def test_antiderivative_vs_derivative(self):
        np.random.seed(1234)
        x = np.linspace(0, 1, 30) ** 2
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0, k=5)
        spp = scipy.interpolate.PPoly.from_spline(spl)
        pp = PPoly(spp.c, spp.x)

        for dx in range(0, 10):
            ipp = pp.antiderivative(dx)

            # check that derivative is inverse op
            pp2 = ipp.derivative(dx)
            assert_allclose(pp.c, pp2.c)

            # check continuity
            for k in range(dx):
                pp2 = ipp.derivative(k)

                r = 1e-13
                endpoint = r * pp2.x[:-1] + (1 - r) * pp2.x[1:]

                assert_allclose(
                    pp2(pp2.x[1:]),
                    pp2(endpoint),
                    rtol=1e-7,
                    err_msg="dx=%d k=%d" % (dx, k),
                )

    def test_antiderivative_continuity(self):
        c = np.array([[2, 1, 2, 2], [2, 1, 3, 3]]).T
        x = np.array([0, 0.5, 1])

        p = PPoly(c, x)
        ip = p.antiderivative()

        # check continuity
        assert_allclose(ip(0.5 - 1e-9), ip(0.5 + 1e-9), rtol=1e-8)

        # check that only lowest order coefficients were changed
        p2 = ip.derivative()
        assert_allclose(p2.c, p.c)

    def test_integrate(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0, k=5)
        spp = scipy.interpolate.PPoly.from_spline(spl)
        pp = PPoly(spp.c, spp.x)

        a, b = 0.3, 0.9
        ig = pp.integrate(a, b)

        ipp = pp.antiderivative()
        assert_allclose(ig, ipp(b) - ipp(a))
        assert_allclose(ig, splint(a, b, spl))

        a, b = -0.3, 0.9
        ig = pp.integrate(a, b, extrapolate=True)
        assert_allclose(ig, ipp(b) - ipp(a))

        assert_(np.isnan(pp.integrate(a, b, extrapolate=False)).all())

    def test_integrate_readonly(self):
        x = np.array([1, 2, 4])
        c = np.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])

        for writeable in (True, False):
            x.flags.writeable = writeable

            P = PPoly(c, x)
            vals = P.integrate(1, 4)

            assert_(np.isfinite(vals).all())

    def test_integrate_periodic(self):
        x = np.array([1, 2, 4])
        c = np.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])

        P = PPoly(c, x, extrapolate="periodic")
        I = P.antiderivative()

        period_int = I(4) - I(1)

        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)

        assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
        assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(
            P.integrate(3.5, 5 + 12), I(2) - I(1) + I(4) - I(3.5) + 4 * period_int
        )

        assert_allclose(P.integrate(0, -1), I(2) - I(3))
        assert_allclose(P.integrate(-9, -10), I(2) - I(3))
        assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)

    def test_extrapolate_attr(self):
        # 1 - x**2
        c = np.array([[-1, 0, 1]]).T
        x = np.array([0, 1])
        x1 = 1 / 2

        for extrapolate in [True, False, None]:
            pp = PPoly(c, x, extrapolate=extrapolate)
            pp_d = pp.derivative()
            pp_i = pp.antiderivative()

            if extrapolate is False:
                assert_(np.isnan(pp([-0.1, 1.1])).all())
                assert_(np.isnan(pp_i([-0.1, 1.1])).all())
                assert_(np.isnan(pp_d([-0.1, 1.1])).all())
            else:
                assert_allclose(pp([-0.1, 1.1]), [1 - 0.1**2, 1 - 1.1**2])
                assert_(not np.isnan(pp_i([-0.1, 1.1])).any())
                assert_(not np.isnan(pp_d([-0.1, 1.1])).any())

            # extra test for gh#85
            assert pp(x1).shape == ()
            assert pp_d(x1).shape == ()
            assert pp_i(x1).shape == ()


def _ppoly_eval_1(c, x, xps):
    """Evaluate piecewise polynomial manually."""
    out = np.zeros((len(xps), c.shape[2]))
    for i, xp in enumerate(xps):
        if xp < 0 or xp > 1:
            out[i, :] = np.nan
            continue
        j = np.searchsorted(x, xp) - 1
        d = xp - x[j]
        assert_(x[j] <= xp < x[j + 1])
        r = sum(c[k, j] * d ** (c.shape[0] - k - 1) for k in range(c.shape[0]))
        out[i, :] = r
    return out


def _ppoly_eval_2(coeffs, breaks, xnew, fill=np.nan):
    """Evaluate piecewise polynomial manually (another way)."""
    a = breaks[0]
    b = breaks[-1]
    K = coeffs.shape[0]

    saveshape = np.shape(xnew)
    xnew = np.ravel(xnew)
    res = np.empty_like(xnew)
    mask = (xnew >= a) & (xnew <= b)
    res[~mask] = fill
    xx = xnew.compress(mask)
    indxs = np.searchsorted(breaks, xx) - 1
    indxs = indxs.clip(0, len(breaks))
    pp = coeffs
    diff = xx - breaks.take(indxs)
    V = np.vander(diff, N=K)
    values = np.array([np.dot(V[k, :], pp[:, indxs[k]]) for k in range(len(xx))])
    res[mask] = values
    res.shape = saveshape
    return res


class TestPCHIP:
    def _make_random(self, npts=20):
        np.random.seed(1234)
        xi = np.sort(np.random.random(npts))
        yi = np.random.random(npts)
        return PchipInterpolator(xi, yi), xi, yi

    def test_overshoot(self):
        # PCHIP should not overshoot
        p, xi, yi = self._make_random()
        for i in range(len(xi) - 1):
            x1, x2 = xi[i], xi[i + 1]
            y1, y2 = yi[i], yi[i + 1]
            if y1 > y2:
                y1, y2 = y2, y1
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert_(((y1 <= yp + 1e-15) & (yp <= y2 + 1e-15)).all())

    def test_monotone(self):
        # PCHIP should preserve monotonicty
        p, xi, yi = self._make_random()
        for i in range(len(xi) - 1):
            x1, x2 = xi[i], xi[i + 1]
            y1, y2 = yi[i], yi[i + 1]
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert_(((y2 - y1) * (yp[1:] - yp[:1]) > 0).all())

    def test_cast(self):
        # regression test for integer input data, see gh-3453
        data = np.array(
            [
                [0, 4, 12, 27, 47, 60, 79, 87, 99, 100],
                [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55],
            ]
        )
        xx = np.arange(100)
        curve = PchipInterpolator(data[0], data[1])(xx)

        data1 = data * 1.0
        curve1 = PchipInterpolator(data1[0], data1[1])(xx)

        assert_allclose(curve, curve1, atol=1e-14, rtol=1e-14)

    def test_nag(self):
        # Example from NAG C implementation,
        # http://nag.com/numeric/cl/nagdoc_cl25/html/e01/e01bec.html
        # suggested in gh-5326 as a smoke test for the way the derivatives
        # are computed (see also gh-3453)
        dataStr = """
          7.99   0.00000E+0
          8.09   0.27643E-4
          8.19   0.43750E-1
          8.70   0.16918E+0
          9.20   0.46943E+0
         10.00   0.94374E+0
         12.00   0.99864E+0
         15.00   0.99992E+0
         20.00   0.99999E+0
        """
        data = np.loadtxt(io.StringIO(dataStr))
        pch = PchipInterpolator(data[:, 0], data[:, 1])

        resultStr = """
           7.9900       0.0000
           9.1910       0.4640
          10.3920       0.9645
          11.5930       0.9965
          12.7940       0.9992
          13.9950       0.9998
          15.1960       0.9999
          16.3970       1.0000
          17.5980       1.0000
          18.7990       1.0000
          20.0000       1.0000
        """
        result = np.loadtxt(io.StringIO(resultStr))
        assert_allclose(result[:, 1], pch(result[:, 0]), rtol=0.0, atol=5e-5)

    def test_endslopes(self):
        # this is a smoke test for gh-3453: PCHIP interpolator should not
        # set edge slopes to zero if the data do not suggest zero edge derivatives
        x = np.array([0.0, 0.1, 0.25, 0.35])
        y1 = np.array([279.35, 0.5e3, 1.0e3, 2.5e3])
        y2 = np.array([279.35, 2.5e3, 1.50e3, 1.0e3])
        for pp in (PchipInterpolator(x, y1), PchipInterpolator(x, y2)):
            for t in (x[0], x[-1]):
                assert_(pp(t, 1) != 0)

    def test_all_zeros(self):
        x = np.arange(10)
        y = np.zeros_like(x)

        # this should work and not generate any warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            pch = PchipInterpolator(x, y)

        xx = np.linspace(0, 9, 101)
        assert_array_equal(pch(xx), 0.0)

    def test_two_points(self):
        # regression test for gh-6222: PchipInterpolator([0, 1], [0, 1]) fails because
        # it tries to use a three-point scheme to estimate edge derivatives,
        # while there are only two points available.
        # Instead, it should construct a linear interpolator.
        x = np.linspace(0, 1, 11)
        p = PchipInterpolator([0, 1], [0, 2])
        assert_allclose(p(x), 2 * x, atol=1e-15)

    def test_PchipInterpolator(self):
        assert_array_almost_equal(
            PchipInterpolator([1, 2, 3], [4, 5, 6])([0.5], nu=1), [1.0]
        )

        assert_array_almost_equal(
            PchipInterpolator([1, 2, 3], [4, 5, 6])([0.5], nu=0), [3.5]
        )


class TestCubicSpline:
    @staticmethod
    def check_correctness(S, bc_start="not-a-knot", bc_end="not-a-knot", tol=1e-14):
        """Check that spline coefficients satisfy the continuity and bc."""
        x = S.x
        c = S.c
        dx = np.diff(x)
        dx = dx.reshape([dx.shape[0]] + [1] * (c.ndim - 2))
        dxi = dx[:-1]

        # Check C2 continuity.
        assert_allclose(
            c[3, 1:],
            c[0, :-1] * dxi**3 + c[1, :-1] * dxi**2 + c[2, :-1] * dxi + c[3, :-1],
            rtol=tol,
            atol=tol,
        )
        assert_allclose(
            c[2, 1:],
            3 * c[0, :-1] * dxi**2 + 2 * c[1, :-1] * dxi + c[2, :-1],
            rtol=tol,
            atol=tol,
        )
        assert_allclose(c[1, 1:], 3 * c[0, :-1] * dxi + c[1, :-1], rtol=tol, atol=tol)

        # Check that we found a parabola, the third derivative is 0.
        if x.size == 3 and bc_start == "not-a-knot" and bc_end == "not-a-knot":
            assert_allclose(c[0], 0, rtol=tol, atol=tol)
            return

        # Check periodic boundary conditions.
        if bc_start == "periodic":
            assert_allclose(S(x[0], 0), S(x[-1], 0), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 1), S(x[-1], 1), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 2), S(x[-1], 2), rtol=tol, atol=tol)
            return

        # Check other boundary conditions.
        if bc_start == "not-a-knot":
            if x.size == 2:
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[0], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, 0], c[0, 1], rtol=tol, atol=tol)
        elif bc_start == "clamped":
            assert_allclose(S(x[0], 1), 0, rtol=tol, atol=tol)
        elif bc_start == "natural":
            assert_allclose(S(x[0], 2), 0, rtol=tol, atol=tol)
        else:
            order, value = bc_start
            assert_allclose(S(x[0], order), value, rtol=tol, atol=tol)

        if bc_end == "not-a-knot":
            if x.size == 2:
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[1], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, -1], c[0, -2], rtol=tol, atol=tol)
        elif bc_end == "clamped":
            assert_allclose(S(x[-1], 1), 0, rtol=tol, atol=tol)
        elif bc_end == "natural":
            assert_allclose(S(x[-1], 2), 0, rtol=2 * tol, atol=2 * tol)
        else:
            order, value = bc_end
            assert_allclose(S(x[-1], order), value, rtol=tol, atol=tol)

    def check_all_bc(self, x, y, axis):
        deriv_shape = list(y.shape)
        del deriv_shape[axis]
        first_deriv = np.empty(deriv_shape)
        first_deriv.fill(2)
        second_deriv = np.empty(deriv_shape)
        second_deriv.fill(-1)
        bc_all = [
            "not-a-knot",
            "natural",
            "clamped",
            (1, first_deriv),
            (2, second_deriv),
        ]
        for bc in bc_all[:3]:
            S = CubicSpline(x, y, axis=axis, bc_type=bc)
            self.check_correctness(S, bc, bc)

        for bc_start in bc_all:
            for bc_end in bc_all:
                S = CubicSpline(x, y, axis=axis, bc_type=(bc_start, bc_end))
                self.check_correctness(S, bc_start, bc_end, tol=2e-14)

    def test_general(self):
        x = np.array([-1, 0, 0.5, 2, 4, 4.5, 5.5, 9])
        y = np.array([0, -0.5, 2, 3, 2.5, 1, 1, 0.5])
        for n in [2, 3, x.size]:
            self.check_all_bc(x[:n], y[:n], 0)

            Y = np.empty((2, n, 2))
            Y[0, :, 0] = y[:n]
            Y[0, :, 1] = y[:n] - 1
            Y[1, :, 0] = y[:n] + 2
            Y[1, :, 1] = y[:n] + 3
            self.check_all_bc(x[:n], Y, 1)

    def test_dtypes(self):
        x = np.array([0, 1, 2, 3], dtype=int)
        y = np.array([-5, 2, 3, 1], dtype=int)
        S = CubicSpline(x, y)
        self.check_correctness(S)

        y = np.array([-1 + 1j, 0.0, 1 - 1j, 0.5 - 1.5j])
        S = CubicSpline(x, y)
        self.check_correctness(S)

        S = CubicSpline(x, x**3, bc_type=("natural", (1, 2j)))
        self.check_correctness(S, "natural", (1, 2j))

        y = np.array([-5, 2, 3, 1])
        S = CubicSpline(x, y, bc_type=[(1, 2 + 0.5j), (2, 0.5 - 1j)])
        self.check_correctness(S, (1, 2 + 0.5j), (2, 0.5 - 1j))

    def test_incorrect_inputs(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])
        xc = np.array([1 + 1j, 2, 3, 4])
        xn = np.array([np.nan, 2, 3, 4])
        xo = np.array([2, 1, 3, 4])
        yn = np.array([np.nan, 2, 3, 4])
        y3 = [1, 2, 3]
        x1 = [1]
        y1 = [1]

        assert_raises(ValueError, CubicSpline, xc, y)
        assert_raises(ValueError, CubicSpline, xn, y)
        assert_raises(ValueError, CubicSpline, x, yn)
        assert_raises(ValueError, CubicSpline, xo, y)
        assert_raises(ValueError, CubicSpline, x, y3)
        assert_raises(ValueError, CubicSpline, x[:, np.newaxis], y)
        assert_raises(ValueError, CubicSpline, x1, y1)

        wrong_bc = [
            ("periodic", "clamped"),
            ((2, 0), (3, 10)),
            ((1, 0),),
            (0.0, 0.0),
            "not-a-typo",
        ]

        for bc_type in wrong_bc:
            assert_raises(ValueError, CubicSpline, x, y, 0, bc_type, True)

        # Shapes mismatch when giving arbitrary derivative values:
        Y = np.c_[y, y]
        bc1 = ("clamped", (1, 0))
        bc2 = ("clamped", (1, [0, 0, 0]))
        bc3 = ("clamped", (1, [[0, 0]]))
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc1, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc2, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc3, True)


def test_CubicHermiteSpline_correctness():
    x = [0, 2, 7]
    y = [-1, 2, 3]
    dydx = [0, 3, 7]
    s = CubicHermiteSpline(x, y, dydx)
    assert_allclose(s(x), y, rtol=1e-15)
    assert_allclose(s(x, 1), dydx, rtol=1e-15)


def test_CubicHermiteSpline_error_handling():
    x = [1, 2, 3]
    y = [0, 3, 5]
    dydx = [1, -1, 2, 3]
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx)

    dydx_with_nan = [1, 0, np.nan]
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx_with_nan)
