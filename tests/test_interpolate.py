"""Tests for interpolation functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jax_config

from interpax import (
    Interpolator1D,
    Interpolator2D,
    Interpolator3D,
    InterpolatorNd,
    fft_interp1d,
    fft_interp2d,
    interp1d,
    interp2d,
    interp3d,
    interpNd,
)

jax_config.update("jax_enable_x64", True)


class TestInterp1D:
    """Tests for interp1d function."""

    @pytest.mark.unit
    @pytest.mark.parametrize("x", [np.linspace(0, 2 * np.pi, 10000), 0.0])
    @pytest.mark.parametrize(
        "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
    )
    def test_interp1d(self, x, dtype):
        """Test accuracy of different 1d interpolation methods."""
        xp = np.linspace(0, 2 * np.pi, 100)
        if jnp.iscomplexobj(dtype):
            f = lambda x: jnp.sin(x) + 1j * jnp.sin(x)
        else:
            f = lambda x: jnp.sin(x)
        fp = f(xp)

        interp1 = lambda xq, *args, **kwargs: interp1d(xq, *args, **kwargs)
        interp2 = lambda xq, *args, **kwargs: Interpolator1D(*args, **kwargs)(xq)

        for interp in [interp1, interp2]:
            fq = interp(x, xp, fp, method="nearest")
            np.testing.assert_allclose(fq, f(x), rtol=1e-2, atol=1e-1)

            fq = interp(x, xp, fp, method="linear")
            np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1e-3)

            fq = interp(x, xp, fp, method="cubic")
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="cubic2")
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="cardinal")
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="catmull-rom")
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="monotonic")
            np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1e-3)

            fq = interp(x, xp, fp, method="monotonic-0")
            np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1.5e-2)

            fq = interp(x, xp, fp, method="akima")
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=2e-5)

    @pytest.mark.unit
    def test_interp1d_vector_valued(self):
        """Test for interpolating vector valued function."""
        xp = np.linspace(0, 2 * np.pi, 100)
        x = np.linspace(0, 2 * np.pi, 300)[10:-10]
        f = lambda x: np.array([np.sin(x), np.cos(x)])
        fp = f(xp).T

        fq = interp1d(x, xp, fp, method="nearest")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-2, atol=1e-1)

        fq = interp1d(x, xp, fp, method="linear")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-4, atol=1e-3)

        fq = interp1d(x, xp, fp, method="cubic")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="cubic2")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="cardinal")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="catmull-rom")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="monotonic")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-4, atol=1e-3)

        fq = interp1d(x, xp, fp, method="monotonic-0")
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-4, atol=1e-2)

    @pytest.mark.unit
    def test_interp1d_extrap_periodic(self):
        """Test extrapolation and periodic BC of 1d interpolation."""
        xp = np.linspace(0, 2 * np.pi, 200)
        x = np.linspace(-1, 2 * np.pi + 1, 10000)
        f = lambda x: np.sin(x)
        fp = f(xp)

        fq = interp1d(x, xp, fp, method="cubic", extrap=False)
        assert np.isnan(fq[0])
        assert np.isnan(fq[-1])

        fq = interp1d(x, xp, fp, method="cubic", extrap=True)
        assert not np.isnan(fq[0])
        assert not np.isnan(fq[-1])

        fq = interp1d(x, xp, fp, method="cubic", period=2 * np.pi)
        np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-2)

    @pytest.mark.unit
    def test_interp1d_monotonic(self):
        """Ensure monotonic interpolation is actually monotonic."""
        # true function is just linear with a jump discontinuity at x=1.5
        x = np.linspace(-4, 5, 10)
        f = np.heaviside(x - 1.5, 0) + 0.1 * x
        xq = np.linspace(-4, 5, 1000)
        dfc = interp1d(xq, x, f, derivative=1, method="cubic")
        dfm = interp1d(xq, x, f, derivative=1, method="monotonic")
        dfm0 = interp1d(xq, x, f, derivative=1, method="monotonic-0")
        assert dfc.min() < 0  # cubic interpolation undershoots, giving negative slope
        assert dfm.min() > 0  # monotonic interpolation doesn't
        assert dfm0.min() >= 0  # monotonic-0 doesn't overshoot either
        # ensure monotonic-0 has 0 slope at end points
        np.testing.assert_allclose(dfm0[np.array([0, -1])], 0, atol=1e-12)


class TestInterp2D:
    """Tests for interp2d function."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "x, y",
        [
            (np.linspace(0, 3 * np.pi, 1000), np.linspace(0, 2 * np.pi, 1000)),
            (0.0, 0.0),
        ],
    )
    @pytest.mark.parametrize(
        "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
    )
    def test_interp2d(self, x, y, dtype):
        """Test accuracy of different 2d interpolation methods."""
        xp = np.linspace(0, 3 * np.pi, 99)
        yp = np.linspace(0, 2 * np.pi, 40)
        xxp, yyp = np.meshgrid(xp, yp, indexing="ij")

        if jnp.iscomplexobj(dtype):
            f = lambda x, y: jnp.sin(x) * jnp.cos(y) + 1j * jnp.sin(x) * jnp.cos(y)
        else:
            f = lambda x, y: jnp.sin(x) * jnp.cos(y)

        fp = f(xxp, yyp)

        interp1 = lambda xq, yq, *args, **kwargs: interp2d(xq, yq, *args, **kwargs)
        interp2 = lambda xq, yq, *args, **kwargs: Interpolator2D(*args, **kwargs)(
            xq, yq
        )

        for interp in [interp1, interp2]:
            fq = interp(
                x, y, xp, yp, fp, method="nearest", period=(2 * np.pi, 2 * np.pi)
            )
            np.testing.assert_allclose(fq, f(x, y), rtol=1e-2, atol=1)

            fq = interp(
                x, y, xp, yp, fp, method="linear", period=(2 * np.pi, 2 * np.pi)
            )
            np.testing.assert_allclose(fq, f(x, y), rtol=1e-4, atol=1e-2)
            atol = 2e-3
            rtol = 1e-5
            fq = interp(x, y, xp, yp, fp, method="cubic", period=(2 * np.pi, 2 * np.pi))
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

            fq = interp(
                x, y, xp, yp, fp, method="cubic2", period=(2 * np.pi, 2 * np.pi)
            )
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

            fq = interp(
                x, y, xp, yp, fp, method="catmull-rom", period=(2 * np.pi, 2 * np.pi)
            )
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

            fq = interp(
                x, y, xp, yp, fp, method="cardinal", period=(2 * np.pi, 2 * np.pi)
            )
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)
            fq = interp(x, y, xp, yp, fp, method="akima", period=(2 * np.pi, 2 * np.pi))
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)
            fq = interp(
                x, y, xp, yp, fp, method="monotonic", period=(2 * np.pi, 2 * np.pi)
            )
            np.testing.assert_allclose(fq, f(x, y), rtol=1e-2, atol=1e-3)

    @pytest.mark.unit
    def test_interp2d_vector_valued(self):
        """Test for interpolating vector valued function."""
        xp = np.linspace(0, 3 * np.pi, 99)
        yp = np.linspace(0, 2 * np.pi, 40)
        x = np.linspace(0, 3 * np.pi, 200)
        y = np.linspace(0, 2 * np.pi, 200)
        xxp, yyp = np.meshgrid(xp, yp, indexing="ij")

        f = lambda x, y: np.array([np.sin(x) * np.cos(y), np.sin(x) + np.cos(y)])
        fp = f(xxp.T, yyp.T).T

        fq = interp2d(x, y, xp, yp, fp, method="nearest")
        np.testing.assert_allclose(fq, f(x, y).T, rtol=1e-2, atol=1.2e-1)

        fq = interp2d(x, y, xp, yp, fp, method="linear")
        np.testing.assert_allclose(fq, f(x, y).T, rtol=1e-3, atol=1e-2)

        fq = interp2d(x, y, xp, yp, fp, method="cubic")
        np.testing.assert_allclose(fq, f(x, y).T, rtol=1e-5, atol=2e-3)


class TestInterp3D:
    """Tests for interp3d function."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "x, y, z",
        [
            (
                np.linspace(0, np.pi, 1000),
                np.linspace(0, 2 * np.pi, 1000),
                np.linspace(0, 3, 1000),
            ),
            (0.0, 0.0, 0.0),
        ],
    )
    @pytest.mark.parametrize(
        "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
    )
    def test_interp3d(self, x, y, z, dtype):
        """Test accuracy of different 3d interpolation methods."""
        xp = np.linspace(0, np.pi, 20)
        yp = np.linspace(0, 2 * np.pi, 30)
        zp = np.linspace(0, 3, 25)
        xxp, yyp, zzp = np.meshgrid(xp, yp, zp, indexing="ij")

        if jnp.iscomplexobj(dtype):
            f = (
                lambda x, y, z: jnp.sin(x) * jnp.cos(y) * z**2
                + 1j * jnp.sin(x) * jnp.cos(y) * z**2
            )
        else:
            f = lambda x, y, z: jnp.sin(x) * jnp.cos(y) * z**2
        fp = f(xxp, yyp, zzp)

        interp1 = lambda xq, yq, zq, *args, **kwargs: interp3d(
            xq, yq, zq, *args, **kwargs
        )
        interp2 = lambda xq, yq, zq, *args, **kwargs: Interpolator3D(*args, **kwargs)(
            xq, yq, zq
        )

        for interp in [interp1, interp2]:
            fq = interp(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-5, atol=1e-2)

            fq = interp(x, y, z, xp, yp, zp, fp, method="nearest")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-2, atol=1)

            fq = interp(x, y, z, xp, yp, zp, fp, method="linear")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-3, atol=1e-1)

            atol = 5.5e-3
            rtol = 1e-5
            fq = interp(x, y, z, xp, yp, zp, fp, method="cubic")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

            fq = interp(x, y, z, xp, yp, zp, fp, method="cubic2")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

            fq = interp(x, y, z, xp, yp, zp, fp, method="catmull-rom")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

            fq = interp(x, y, z, xp, yp, zp, fp, method="cardinal")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

    @pytest.mark.unit
    def test_interp3d_vector_valued(self):
        """Test for interpolating vector valued function."""
        x = np.linspace(0, np.pi, 1000)
        y = np.linspace(0, 2 * np.pi, 1000)
        z = np.linspace(0, 3, 1000)
        xp = np.linspace(0, np.pi, 20)
        yp = np.linspace(0, 2 * np.pi, 30)
        zp = np.linspace(0, 3, 25)
        xxp, yyp, zzp = np.meshgrid(xp, yp, zp, indexing="ij")

        f = lambda x, y, z: np.array([np.sin(x) * np.cos(y) * z**2, 0.1 * (x + y - z)])
        fp = f(xxp.T, yyp.T, zzp.T).T

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="nearest")
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-2, atol=1)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="linear")
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-3, atol=1e-1)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="cubic")
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-5, atol=5e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
def test_fft_interp1d(dtype):
    """Test for 1d Fourier interpolation."""
    if jnp.iscomplexobj(dtype):
        fun = lambda x: 2 * jnp.sin(1 * x) + 4j * jnp.cos(3 * x) + 1
    else:
        fun = lambda x: 2 * jnp.sin(1 * x) + 4 * jnp.cos(3 * x) + 1

    x = {"o": {}, "e": {}}
    x["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    x["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    x["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    x["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)
    f1 = {}
    for p in ["o", "e"]:
        f1[p] = {}
        for i in [1, 2]:
            f1[p][i] = fun(x[p][i])

    for sp in ["o", "e"]:  # source parity
        fi = f1[sp][1]
        fs = fun(x[sp][1] + 0.2)
        np.testing.assert_allclose(
            fs, fft_interp1d(fi, *fi.shape, sx=0.2, dx=np.diff(x[sp][1])[0]).squeeze()
        )
        for ep in ["o", "e"]:  # eval parity
            for s in ["up", "down"]:  # up or downsample
                if s == "up":
                    xs = 1
                    xe = 2
                else:
                    xs = 2
                    xe = 1
                true = fun(x[ep][xe])
                interp = fft_interp1d(f1[sp][xs], x[ep][xe].size)
                np.testing.assert_allclose(true, interp, atol=1e-12, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
def test_fft_interp2d(dtype):
    """Test for 2d Fourier interpolation."""
    if jnp.iscomplexobj(dtype):
        fun2 = lambda x, y: (
            2j * jnp.sin(1 * x[:, None])
            - 1.2 * jnp.cos(2 * x[:, None])
            + 3 * jnp.cos(3 * y[None])
            - 2j * jnp.cos(5 * y[None])
            + 1
        )
    else:
        fun2 = lambda x, y: (
            2 * jnp.sin(1 * x[:, None])
            - 1.2 * jnp.cos(2 * x[:, None])
            + 3 * jnp.cos(3 * y[None])
            - 2 * jnp.cos(5 * y[None])
            + 1
        )

    x = {"o": {}, "e": {}}
    y = {"o": {}, "e": {}}
    x["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    x["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    x["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    x["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)
    y["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    y["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    y["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    y["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)

    f2 = {}
    for xp in ["o", "e"]:
        f2[xp] = {}
        for yp in ["o", "e"]:
            f2[xp][yp] = {}
            for i in [1, 2]:
                f2[xp][yp][i] = {}
                for j in [1, 2]:
                    f2[xp][yp][i][j] = fun2(x[xp][i], y[yp][j])

    for spx in ["o", "e"]:  # source parity x
        for spy in ["o", "e"]:  # source parity y
            fi = f2[spx][spy][1][1]
            fs = fun2(x[spx][1] + 0.2, y[spy][1] + 0.3)
            np.testing.assert_allclose(
                fs,
                fft_interp2d(
                    fi,
                    *fi.shape,
                    sx=0.2,
                    sy=0.3,
                    dx=np.diff(x[spx][1])[0],
                    dy=np.diff(y[spy][1])[0]
                ).squeeze(),
            )
            for epx in ["o", "e"]:  # eval parity x
                for epy in ["o", "e"]:  # eval parity y
                    for sx in ["up", "down"]:  # up or downsample x
                        if sx == "up":
                            xs = 1
                            xe = 2
                        else:
                            xs = 2
                            xe = 1
                        for sy in ["up", "down"]:  # up or downsample y
                            if sy == "up":
                                ys = 1
                                ye = 2
                            else:
                                ys = 2
                                ye = 1
                            true = fun2(x[epx][xe], y[epy][ye])
                            interp = fft_interp2d(
                                f2[spx][spy][xs][ys], x[epx][xe].size, y[epy][ye].size
                            )
                            np.testing.assert_allclose(
                                true, interp, atol=1e-12, rtol=1e-12
                            )


class TestAD:
    """Tests to make sure JAX transforms work correctly."""

    def _finite_difference(self, f, x, eps=1e-8):
        """Util for 2nd order centered finite differences."""
        x0 = np.atleast_1d(x).squeeze()
        f0 = f(x0)
        m = f0.size
        n = x0.size
        J = np.zeros((m, n))
        h = np.maximum(1.0, np.abs(x0)) * eps
        h_vecs = np.diag(np.atleast_1d(h))
        for i in range(n):
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            if x0.ndim:
                dx = x2[i] - x1[i]
            else:
                dx = x2 - x1
            f1 = f(x1)
            f2 = f(x2)
            df = f2 - f1
            dfdx = df / dx
            J[:, i] = dfdx.flatten()
        if m == 1:
            J = np.ravel(J)
        return J

    @pytest.mark.unit
    def test_ad_interp1d(self):
        """Test AD of different 1d interpolation methods."""
        xp = np.linspace(0, 2 * np.pi, 10)
        x = np.linspace(0, 2 * np.pi, 20)
        f = lambda x: np.sin(x)
        fp = f(xp)

        for method in ["cubic", "cubic2", "cardinal", "monotonic"]:

            interp1 = lambda xp: interp1d(x, xp, fp, method=method)
            interp2 = lambda xp: Interpolator1D(xp, fp, method=method)(x)

            jacf1 = jax.jacfwd(interp1)(xp)
            jacf2 = jax.jacfwd(interp2)(xp)

            jacr1 = jax.jacrev(interp1)(xp)
            jacr2 = jax.jacrev(interp2)(xp)

            jacd1 = self._finite_difference(interp1, xp)
            jacd2 = self._finite_difference(interp2, xp)

            np.testing.assert_allclose(jacf1, jacf2, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(jacr1, jacr2, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(jacf1, jacr1, rtol=1e-14, atol=1e-14)
            # for some reason finite difference gives nan at endpoints so ignore that
            np.testing.assert_allclose(jacf1[1:-1], jacd1[1:-1], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(jacf2[1:-1], jacd2[1:-1], rtol=1e-6, atol=1e-6)

    @pytest.mark.unit
    def test_ad_interp2d(self):
        """Test AD of different 2d interpolation methods."""
        xp = np.linspace(0, 4 * np.pi, 20)
        yp = np.linspace(0, 2 * np.pi, 20)
        y = np.linspace(0, 2 * np.pi, 30)
        x = np.linspace(0, 2 * np.pi, 30)
        xxp, yyp = np.meshgrid(xp, yp, indexing="ij")

        f = lambda x, y: np.sin(x) * np.cos(y)
        fp = f(xxp, yyp)

        for method in ["cubic", "cubic2", "cardinal"]:

            interp1 = lambda xp: interp2d(x, y, xp, yp, fp, method=method)
            interp2 = lambda xp: Interpolator2D(xp, yp, fp, method=method)(x, y)

            jacf1 = jax.jacfwd(interp1)(xp)
            jacf2 = jax.jacfwd(interp2)(xp)

            jacr1 = jax.jacrev(interp1)(xp)
            jacr2 = jax.jacrev(interp2)(xp)

            jacd1 = self._finite_difference(interp1, xp)
            jacd2 = self._finite_difference(interp2, xp)

            np.testing.assert_allclose(jacf1, jacf2, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(jacr1, jacr2, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(jacf1, jacr1, rtol=1e-14, atol=1e-14)
            # for some reason finite difference gives nan at endpoints so ignore that
            np.testing.assert_allclose(jacf1[1:-1], jacd1[1:-1], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(jacf2[1:-1], jacd2[1:-1], rtol=1e-6, atol=1e-6)

    @pytest.mark.unit
    def test_ad_interp3d(self):
        """Test AD of different 3d interpolation methods."""
        xp = np.linspace(0, np.pi, 10)
        yp = np.linspace(0, 2 * np.pi, 15)
        zp = np.linspace(0, 1, 12)
        x = np.linspace(0, np.pi, 13)
        y = np.linspace(0, 2 * np.pi, 13)
        z = np.linspace(0, 1, 13)
        xxp, yyp, zzp = np.meshgrid(xp, yp, zp, indexing="ij")

        f = lambda x, y, z: np.sin(x) * np.cos(y) * z**2
        fp = f(xxp, yyp, zzp)

        for method in ["cubic", "cubic2", "cardinal"]:

            interp1 = lambda xp: interp3d(x, y, z, xp, yp, zp, fp, method=method)
            interp2 = lambda xp: Interpolator3D(xp, yp, zp, fp, method=method)(x, y, z)

            jacf1 = jax.jacfwd(interp1)(xp)
            jacf2 = jax.jacfwd(interp2)(xp)

            jacr1 = jax.jacrev(interp1)(xp)
            jacr2 = jax.jacrev(interp2)(xp)

            jacd1 = self._finite_difference(interp1, xp)
            jacd2 = self._finite_difference(interp2, xp)

            np.testing.assert_allclose(jacf1, jacf2, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(jacr1, jacr2, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(jacf1, jacr1, rtol=1e-12, atol=1e-12)
            # for some reason finite difference gives nan at endpoints so ignore that
            np.testing.assert_allclose(jacf1[1:-1], jacd1[1:-1], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(jacf2[1:-1], jacd2[1:-1], rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_extrap_float():
    """Test for extrap being a float, from gh issue #16."""
    x = jnp.linspace(0, 10, 10)
    y = jnp.linspace(0, 8, 8)
    z = jnp.zeros((10, 8)) + 1.0
    interpol = Interpolator2D(x, y, z, extrap=0.0)
    np.testing.assert_allclose(interpol(4.5, 5.3), 1.0)
    np.testing.assert_allclose(interpol(-4.5, 5.3), 0.0)
    np.testing.assert_allclose(interpol(4.5, -5.3), 0.0)


class TestInterpNd:
    """Tests for N-dimensional interpolation."""

    @pytest.mark.unit
    def test_interpNd_linear_3d(self):
        """Test 3D multilinear interpolation accuracy."""
        x = (
            jnp.linspace(0, 1, 10),
            jnp.linspace(0, 1, 12),
            jnp.linspace(0, 1, 8),
        )
        grid = jnp.meshgrid(*x, indexing="ij")
        f = jnp.sin(2 * np.pi * grid[0]) * jnp.cos(2 * np.pi * grid[1]) * grid[2]

        xq = (jnp.array([0.0, 0.5, 1.0]), jnp.array([0.0, 0.5, 1.0]), jnp.array([0.5, 0.5, 0.5])) 
        fq = interpNd(xq, x, f, method="linear")

        expected = jnp.sin(2 * np.pi * xq[0]) * jnp.cos(2 * np.pi * xq[1]) * xq[2]
        np.testing.assert_allclose(fq, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.unit
    def test_interpNd_linear_4d(self):
        """Test 4D multilinear interpolation."""
        x = tuple(jnp.linspace(0, 1, 5) for _ in range(4))
        grid = jnp.meshgrid(*x, indexing="ij")
        
        f = grid[0] + 2 * grid[1] + 3 * grid[2] + 4 * grid[3]

        xq = tuple(jnp.array([0.25, 0.75]) for _ in range(4))
        fq = interpNd(xq, x, f, method="linear")

        expected = xq[0] + 2 * xq[1] + 3 * xq[2] + 4 * xq[3]
        np.testing.assert_allclose(fq, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_interpNd_nearest(self):
        """Test nearest neighbor interpolation."""
        x = (jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]))
        f = jnp.arange(9).reshape(3, 3).astype(float)

        xq = (jnp.array([0.1, 1.9]), jnp.array([0.1, 1.9]))
        fq = interpNd(xq, x, f, method="nearest")

        np.testing.assert_allclose(fq[0], f[0, 0])  # closest to (0,0)
        np.testing.assert_allclose(fq[1], f[2, 2])  # closest to (2,2)

    @pytest.mark.unit
    def test_interpNd_class(self):
        """Test InterpolatorNd class matches functional API."""
        x = (jnp.linspace(0, 1, 10), jnp.linspace(0, 1, 10), jnp.linspace(0, 1, 10))
        grid = jnp.meshgrid(*x, indexing="ij")
        f = grid[0] * grid[1] * grid[2]

        xq = (jnp.array([0.3, 0.7]), jnp.array([0.4, 0.6]), jnp.array([0.5, 0.5]))

        fq_func = interpNd(xq, x, f, method="linear")
        interp = InterpolatorNd(x, f, method="linear")
        fq_class = interp(*xq)

        np.testing.assert_allclose(fq_func, fq_class, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_interpNd_extrap(self):
        """Test extrapolation behavior."""
        x = (jnp.linspace(0, 1, 5), jnp.linspace(0, 1, 5))
        f = jnp.ones((5, 5))

        xq = (jnp.array([-0.5, 0.5, 1.5]), jnp.array([0.5, 0.5, 0.5]))

        fq = interpNd(xq, x, f, method="linear", extrap=False)
        assert jnp.isnan(fq[0])
        np.testing.assert_allclose(fq[1], 1.0)
        assert jnp.isnan(fq[2])

        fq = interpNd(xq, x, f, method="linear", extrap=True)
        np.testing.assert_allclose(fq, jnp.array([1.0, 1.0, 1.0]), rtol=1e-10)

        fq = interpNd(xq, x, f, method="linear", extrap=0.0)
        np.testing.assert_allclose(fq, jnp.array([0.0, 1.0, 0.0]), rtol=1e-10)

    @pytest.mark.unit
    def test_interpNd_jit(self):
        """Test that interpNd works with jax.jit."""
        x = tuple(jnp.linspace(0, 1, 5) for _ in range(3))
        f = jnp.ones((5, 5, 5))

        @jax.jit
        def interp_fn(xq):
            return interpNd(xq, x, f, method="linear")

        xq = (jnp.array([0.5]), jnp.array([0.5]), jnp.array([0.5]))
        result = interp_fn(xq)
        np.testing.assert_allclose(result, jnp.array([1.0]))

    @pytest.mark.unit
    def test_interpNd_grad(self):
        """Test that gradients work for interpNd."""
        x = tuple(jnp.linspace(0, 1, 5) for _ in range(2))
        grid = jnp.meshgrid(*x, indexing="ij")
        f = grid[0] ** 2 + grid[1] ** 2

        xq = (jnp.array([0.5]), jnp.array([0.5]))

        def loss(f_in):
            return interpNd(xq, x, f_in, method="linear").sum()

        grad_f = jax.grad(loss)(f)

        assert grad_f.shape == f.shape
        assert jnp.any(grad_f != 0)

        eps = 1e-5
        i, j = 2, 2
        f_perturb = f.at[i, j].add(eps)
        fd = (loss(f_perturb) - loss(f)) / eps
        np.testing.assert_allclose(grad_f[i, j], fd, rtol=1e-3)

    @pytest.mark.unit
    def test_interpNd_matches_interp2d_linear(self):
        """Test that interpNd matches interp2d for 2D linear interpolation."""
        x = jnp.linspace(0, 1, 10)
        y = jnp.linspace(0, 1, 12)
        grid = jnp.meshgrid(x, y, indexing="ij")
        f = jnp.sin(grid[0]) * jnp.cos(grid[1])

        xq = jnp.linspace(0.1, 0.9, 5)
        yq = jnp.linspace(0.1, 0.9, 5)

        fq_2d = interp2d(xq, yq, x, y, f, method="linear")
        fq_nd = interpNd((xq, yq), (x, y), f, method="linear")

        np.testing.assert_allclose(fq_2d, fq_nd, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_interpNd_matches_interp3d_linear(self):
        """Test that interpNd matches interp3d for 3D linear interpolation."""
        x = jnp.linspace(0, 1, 8)
        y = jnp.linspace(0, 1, 10)
        z = jnp.linspace(0, 1, 6)
        grid = jnp.meshgrid(x, y, z, indexing="ij")
        f = grid[0] * grid[1] * grid[2]

        xq = jnp.linspace(0.1, 0.9, 3)
        yq = jnp.linspace(0.1, 0.9, 3)
        zq = jnp.linspace(0.1, 0.9, 3)

        fq_3d = interp3d(xq, yq, zq, x, y, z, f, method="linear")
        fq_nd = interpNd((xq, yq, zq), (x, y, z), f, method="linear")

        np.testing.assert_allclose(fq_3d, fq_nd, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_interpNd_invalid_method(self):
        """Test that interpNd raises error for unsupported method."""
        x = (jnp.linspace(0, 1, 5), jnp.linspace(0, 1, 5))
        f = jnp.ones((5, 5))
        xq = (jnp.array([0.5]), jnp.array([0.5]))
        with pytest.raises(ValueError):
            interpNd(xq, x, f, method="cubic")
