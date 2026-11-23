"""Tests for interpolation functions."""

import functools
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jax_config

from interpax import (
    CubicHermiteSpline,
    CubicSmoothingSpline,
    Interpolator1D,
    Interpolator2D,
    Interpolator3D,
    approx_df,
    fft_interp1d,
    fft_interp2d,
    interp1d,
    interp2d,
    interp3d,
)
from interpax._spline import _find_cell

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
        xp = np.real(np.linspace(0, 2 * np.pi, 100).astype(dtype))
        if jnp.iscomplexobj(dtype):
            f = lambda x: jnp.sin(x) + 1j * jnp.sin(x)
        else:
            f = lambda x: jnp.sin(x)
        fp = f(xp)

        if isinstance(x, np.ndarray):
            x = np.real(x.astype(dtype))
        fp = fp.astype(dtype)

        interp1 = lambda xq, *args, **kwargs: interp1d(xq, *args, **kwargs)
        interp2 = lambda xq, *args, **kwargs: Interpolator1D(*args, **kwargs)(xq)

        for interp in [interp1, interp2]:
            fq = interp(x, xp, fp, method="nearest")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-2, atol=1e-1)

            fq = interp(x, xp, fp, method="linear")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1e-3)

            fq = interp(x, xp, fp, method="cubic")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="cubic2")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="cardinal")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="catmull-rom")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

            fq = interp(x, xp, fp, method="monotonic")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1e-3)

            fq = interp(x, xp, fp, method="monotonic-0")
            assert fq.dtype == np.result_type(x, xp, fp)
            np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1.5e-2)

            fq = interp(x, xp, fp, method="akima")
            assert fq.dtype == np.result_type(x, xp, fp)
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
    def test_interp1d_even_spaced(self):
        """Test for interpolating with an evenly spaced grid."""
        xp = np.linspace(0, 2 * np.pi, 100)
        x = np.linspace(0, 2 * np.pi, 300)[10:-10]
        f = lambda x: np.array([np.sin(x), np.cos(x)])
        fp = f(xp).T

        nearest = fp[np.argmin(np.abs(x[:, None] - xp[None]), axis=1)]
        for even_spacing in [True, False]:
            fq = interp1d(x, xp, fp, method="nearest", even_spacing=even_spacing)
            np.testing.assert_allclose(fq, nearest)

        fq = interp1d(x, xp, fp, method="linear", even_spacing=True)
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-4, atol=1e-3)

        fq = interp1d(x, xp, fp, method="cubic", even_spacing=True)
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="cubic2", even_spacing=True)
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="cardinal", even_spacing=True)
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="catmull-rom", even_spacing=True)
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="monotonic", even_spacing=True)
        np.testing.assert_allclose(fq, f(x).T, rtol=1e-4, atol=1e-3)

        fq = interp1d(x, xp, fp, method="monotonic-0", even_spacing=True)
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
        xp = np.real(np.linspace(0, 3 * np.pi, 99).astype(dtype))
        yp = np.real(np.linspace(0, 2 * np.pi, 40).astype(dtype))
        xxp, yyp = np.meshgrid(xp, yp, indexing="ij")

        if jnp.iscomplexobj(dtype):
            f = lambda x, y: jnp.sin(x) * jnp.cos(y) + 1j * jnp.sin(x) * jnp.cos(y)
        else:
            f = lambda x, y: jnp.sin(x) * jnp.cos(y)

        fp = f(xxp, yyp)

        if isinstance(x, np.ndarray):
            x = np.real(x.astype(dtype))
        if isinstance(y, np.ndarray):
            y = np.real(y.astype(dtype))
        fp = fp.astype(dtype)

        interp1 = lambda xq, yq, *args, **kwargs: interp2d(xq, yq, *args, **kwargs)
        interp2 = lambda xq, yq, *args, **kwargs: Interpolator2D(*args, **kwargs)(
            xq, yq
        )

        for interp in [interp1, interp2]:
            fq = interp(
                x, y, xp, yp, fp, method="nearest", period=(2 * np.pi, 2 * np.pi)
            )
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=1e-2, atol=1)
            fq = interp(
                x, y, xp, yp, fp, method="linear", period=(2 * np.pi, 2 * np.pi)
            )
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=1e-4, atol=1e-2)
            atol = 2e-3
            rtol = 1e-5
            fq = interp(x, y, xp, yp, fp, method="cubic", period=(2 * np.pi, 2 * np.pi))
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

            fq = interp(
                x, y, xp, yp, fp, method="cubic2", period=(2 * np.pi, 2 * np.pi)
            )
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

            fq = interp(
                x, y, xp, yp, fp, method="catmull-rom", period=(2 * np.pi, 2 * np.pi)
            )
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

            fq = interp(
                x, y, xp, yp, fp, method="cardinal", period=(2 * np.pi, 2 * np.pi)
            )
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)
            fq = interp(x, y, xp, yp, fp, method="akima", period=(2 * np.pi, 2 * np.pi))
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)
            fq = interp(
                x, y, xp, yp, fp, method="monotonic", period=(2 * np.pi, 2 * np.pi)
            )
            assert fq.dtype == np.result_type(x, y, xp, yp, fp)
            np.testing.assert_allclose(fq, f(x, y), rtol=1e-2, atol=1e-3)
            fq = interp(
                x, y, xp, yp, fp, method="monotonic-0", period=(2 * np.pi, 2 * np.pi)
            )
            np.testing.assert_allclose(fq, f(x, y), rtol=1e-2, atol=2e-2)

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

    @pytest.mark.unit
    def test_interp2d_even_spaced(self):
        """Test for interpolating with evenly spaced grid."""
        xp = np.linspace(0, 3 * np.pi, 99)
        yp = np.linspace(0, 2 * np.pi, 40)
        x = np.linspace(0, 3 * np.pi, 200)
        y = np.linspace(0, 2 * np.pi, 200)
        xxp, yyp = np.meshgrid(xp, yp, indexing="ij")

        f = lambda x, y: np.array([np.sin(x) * np.cos(y), np.sin(x) + np.cos(y)])
        fp = f(xxp.T, yyp.T).T

        fq = interp2d(x, y, xp, yp, fp, method="nearest", even_spacing=True)
        np.testing.assert_allclose(fq, f(x, y).T, rtol=1e-2, atol=1.2e-1)

        fq = interp2d(x, y, xp, yp, fp, method="linear", even_spacing=True)
        np.testing.assert_allclose(fq, f(x, y).T, rtol=1e-3, atol=1e-2)

        fq = interp2d(x, y, xp, yp, fp, method="cubic", even_spacing=True)
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
        xp = np.real(np.linspace(0, np.pi, 20).astype(dtype))
        yp = np.real(np.linspace(0, 2 * np.pi, 30).astype(dtype))
        zp = np.real(np.linspace(0, 3, 25).astype(dtype))
        xxp, yyp, zzp = np.meshgrid(xp, yp, zp, indexing="ij")

        if jnp.iscomplexobj(dtype):
            f = lambda x, y, z: (
                jnp.sin(x) * jnp.cos(y) * z**2 + 1j * jnp.sin(x) * jnp.cos(y) * z**2
            )
        else:
            f = lambda x, y, z: jnp.sin(x) * jnp.cos(y) * z**2
        fp = f(xxp, yyp, zzp)

        if isinstance(x, np.ndarray):
            x = np.real(x.astype(dtype))
        if isinstance(y, np.ndarray):
            y = np.real(y.astype(dtype))
        if isinstance(z, np.ndarray):
            z = np.real(z.astype(dtype))
        fp = fp.astype(dtype)

        interp1 = lambda xq, yq, zq, *args, **kwargs: interp3d(
            xq, yq, zq, *args, **kwargs
        )
        interp2 = lambda xq, yq, zq, *args, **kwargs: Interpolator3D(*args, **kwargs)(
            xq, yq, zq
        )

        for interp in [interp1, interp2]:
            fq = interp(x, y, z, xp, yp, zp, fp)
            assert fq.dtype == np.result_type(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-5, atol=1e-2)

            fq = interp(x, y, z, xp, yp, zp, fp, method="nearest")
            assert fq.dtype == np.result_type(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-2, atol=1)

            fq = interp(x, y, z, xp, yp, zp, fp, method="linear")
            assert fq.dtype == np.result_type(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-3, atol=1e-1)

            atol = 5.5e-3
            rtol = 1e-5
            fq = interp(x, y, z, xp, yp, zp, fp, method="cubic")
            assert fq.dtype == np.result_type(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

            fq = interp(x, y, z, xp, yp, zp, fp, method="cubic2")
            assert fq.dtype == np.result_type(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

            fq = interp(x, y, z, xp, yp, zp, fp, method="catmull-rom")
            assert fq.dtype == np.result_type(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

            fq = interp(x, y, z, xp, yp, zp, fp, method="cardinal")
            assert fq.dtype == np.result_type(x, y, z, xp, yp, zp, fp)
            np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

            fq = interp(x, y, z, xp, yp, zp, fp, method="monotonic")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-2, atol=1e-2)

            fq = interp(x, y, z, xp, yp, zp, fp, method="monotonic-0")
            np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-2, atol=0.3)

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

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="monotonic")
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-2, atol=1e-2)

    @pytest.mark.unit
    def test_interp3d_even_spaced(self):
        """Test for interpolating with evenly spaced grid."""
        x = np.linspace(0, np.pi, 1000)
        y = np.linspace(0, 2 * np.pi, 1000)
        z = np.linspace(0, 3, 1000)
        xp = np.linspace(0, np.pi, 20)
        yp = np.linspace(0, 2 * np.pi, 30)
        zp = np.linspace(0, 3, 25)
        xxp, yyp, zzp = np.meshgrid(xp, yp, zp, indexing="ij")

        f = lambda x, y, z: np.array([np.sin(x) * np.cos(y) * z**2, 0.1 * (x + y - z)])
        fp = f(xxp.T, yyp.T, zzp.T).T

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="nearest", even_spacing=True)
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-2, atol=1)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="linear", even_spacing=True)
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-3, atol=1e-1)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="cubic", even_spacing=True)
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-5, atol=5e-3)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="monotonic", even_spacing=True)
        np.testing.assert_allclose(fq, f(x, y, z).T, rtol=1e-2, atol=1e-2)


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
            f1[p][i] = fun(x[p][i]).astype(dtype)

    for sp in ["o", "e"]:  # source parity
        fi = f1[sp][1].astype(dtype)
        fs = fun(x[sp][1] + 0.2).astype(dtype)
        f_ = fft_interp1d(fi, *fi.shape, sx=0.2, dx=np.diff(x[sp][1])[0]).squeeze()
        np.testing.assert_allclose(
            fs,
            f_,
            rtol=float(1e2 * np.finfo(dtype).eps),
            atol=float(1e2 * np.finfo(dtype).eps),
        )
        for ep in ["o", "e"]:  # eval parity
            for s in ["up", "down"]:  # up or downsample
                if s == "up":
                    xs = 1
                    xe = 2
                else:
                    xs = 2
                    xe = 1
                true = fun(x[ep][xe]).astype(dtype)
                interp = fft_interp1d(f1[sp][xs], x[ep][xe].size)
                np.testing.assert_allclose(
                    true,
                    interp,
                    rtol=float(1e2 * np.finfo(dtype).eps),
                    atol=float(1e2 * np.finfo(dtype).eps),
                )


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
                    f2[xp][yp][i][j] = fun2(x[xp][i], y[yp][j]).astype(dtype)

    for spx in ["o", "e"]:  # source parity x
        for spy in ["o", "e"]:  # source parity y
            fi = f2[spx][spy][1][1].astype(dtype)
            fs = fun2(x[spx][1] + 0.2, y[spy][1] + 0.3).astype(dtype)
            f_ = fft_interp2d(
                fi,
                *fi.shape,
                sx=0.2,
                sy=0.3,
                dx=np.diff(x[spx][1])[0],
                dy=np.diff(y[spy][1])[0],
            ).squeeze()
            np.testing.assert_allclose(
                fs,
                f_,
                rtol=float(1e2 * np.finfo(dtype).eps),
                atol=float(1e2 * np.finfo(dtype).eps),
            )
            assert f_.dtype == dtype
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
                            true = fun2(x[epx][xe], y[epy][ye]).astype(dtype)
                            interp = fft_interp2d(
                                f2[spx][spy][xs][ys], x[epx][xe].size, y[epy][ye].size
                            )
                            np.testing.assert_allclose(
                                true,
                                interp,
                                rtol=float(1e2 * np.finfo(dtype).eps),
                                atol=float(1e2 * np.finfo(dtype).eps),
                            )
                            assert interp.dtype == dtype


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "fun, cls, axis, method, period",
    [
        (interp1d, Interpolator1D, -1, "cubic2", 1.1),
        (interp2d, Interpolator2D, (2, 0), "linear", None),
        (interp2d, Interpolator2D, (2, 0), "cubic", (None, 1.1)),
        (interp3d, Interpolator3D, (3, 1, 0), "cubic", (1.1, None, 1.1)),
    ],
)
def test_interp_axis(fun, cls, axis, method, period):
    """Test interpolating along non-leading axes, and functions matching classes."""
    rng = np.random.default_rng(0)
    axis_ = np.atleast_1d(axis) % (np.size(axis) + 2)
    n = len(axis_)
    grids = [np.linspace(0, 1, 6 + k) for k in range(n)]
    shape = [3, 2]
    for k in np.argsort(axis_):
        shape.insert(axis_[k], len(grids[k]))
    f = rng.random(shape)
    q = [rng.random(5) for _ in range(n)]
    derivative = (1,) + (0,) * (n - 1)
    d = derivative if n > 1 else 1

    # interpolate along the leading axes, then move the query axes where expected
    ref = fun(*q, *grids, np.moveaxis(f, axis_, range(n)), method, d, period=period)
    ref = np.moveaxis(ref, 0, min(axis_))

    out1 = fun(*q, *grids, f, method, d, period=period, axis=axis)
    interp = cls(*grids, f, method, period=period, axis=axis)
    out2 = interp(*q, *derivative)
    # derivatives given back in the layout of f reproduce the same interpolant
    kw = interp.derivs if n > 1 else {}
    out3 = cls(*grids, f, method, period=period, axis=axis, **kw)(*q, *derivative)
    rest = [s for k, s in enumerate(f.shape) if k not in axis_]
    assert out1.shape == tuple(rest[: min(axis_)] + [5] + rest[min(axis_) :])
    np.testing.assert_allclose(out1, ref)
    np.testing.assert_allclose(out2, ref)
    np.testing.assert_allclose(out3, ref)

    with pytest.raises(ValueError):
        fun(*q, *grids, f, method, axis=(0,) * n)


@pytest.mark.unit
def test_import_does_not_initialize_jax_backend():
    """Importing interpax must not trigger JAX backend initialization."""
    script = """
import jax
jax.config.update('jax_enable_x64', True)
import interpax  # noqa: F401
import jax._src.xla_bridge as xb
if xb._backends:
    raise SystemExit(f"JAX backend initialized on import: {list(xb._backends)}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"JAX backend was initialized during `import interpax`.\n"
        f"stderr: {result.stderr}"
    )


@pytest.mark.unit
def test_find_cell():
    """Test cell lookup and detection of evenly spaced knots."""
    rng = np.random.default_rng(0)
    n = 500
    s = np.linspace(0, 1, n)
    grids = {
        # far from the origin, so the first interval alone is a poor spacing estimate
        "offset_f32": (np.linspace(1000, 1001, n, dtype=np.float32), True),
        "nonuniform": (s**2, False),
        # every knot moved by up to 0.3 spacings, beyond what the direct lookup allows
        "perturbed": (s + rng.uniform(-0.3, 0.3, n) / (n - 1), False),
    }
    for name, (x, uniform) in grids.items():
        x = jnp.asarray(x)
        # knots, midpoints, and random points including some outside the range
        u = jnp.asarray(rng.uniform(-0.1, 1.1, 1000), x.dtype)
        xq = jnp.concatenate([x, (x[1:] + x[:-1]) / 2, x[0] + (x[-1] - x[0]) * u])
        ref = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, n - 1)
        for even in [None, False] + [True] * uniform:
            np.testing.assert_array_equal(_find_cell(x, xq, even), ref, err_msg=name)
        assert Interpolator1D(x, x, method="linear").even_spacing == uniform, name

    x = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    assert Interpolator1D(x, x, period=2 * np.pi).even_spacing is True
    y = np.linspace(0, 1, 30) ** 2
    f = np.ones((20, 30))
    assert Interpolator2D(x, y, f, period=(2 * np.pi, None)).even_spacing == (
        True,
        False,
    )
    interp = jax.jit(lambda x: Interpolator1D(x, x, method="linear"))(x)
    assert interp.even_spacing is None


def test_cubic_smoothing_spline():
    """Test CubicSmoothingSpline against scipy and its limiting cases."""
    from scipy.interpolate import make_smoothing_spline

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 3, 30))
    y = np.sin(2 * x) + 0.1 * rng.normal(size=30)
    w = rng.uniform(0.5, 2, 30)
    xq = np.linspace(0, 3, 200)
    Y = np.stack([y, 2 * y], axis=0)

    # p is normalized by the knot spacing and weights before entering the objective
    span = np.ptp(x)
    eff_x = 1 + span**2 / np.sum(np.diff(x) ** 2)
    eff_w = np.sum(w) ** 2 / np.sum(w**2)
    k = 80 * span**3 * x.size**-2 * eff_x**-0.5 * eff_w**-0.5
    for p in [0.01, 0.5, 0.99]:
        pk = p / (p + (1 - p) * k)
        ref = make_smoothing_spline(x, y, w=w, lam=(1 - pk) / pk)(xq)
        out = CubicSmoothingSpline(x, Y, p=p, w=w, axis=1)(xq)
        np.testing.assert_allclose(out, np.stack([ref, 2 * ref]), atol=1e-8)

    line = np.polyval(np.polyfit(x, y, 1, w=np.sqrt(w)), xq)
    np.testing.assert_allclose(
        CubicSmoothingSpline(x, y, p=0, w=w)(xq), line, atol=1e-10
    )
    np.testing.assert_allclose(CubicSmoothingSpline(x, y, p=1)(x), y, atol=1e-10)
    # with incompatible clamped slopes the smoothest admissible spline is a parabola
    par = 5 * (x - x[0]) - (x - x[0]) ** 2 / np.ptp(x)
    par += np.sum(w * (y - par)) / np.sum(w)
    s = CubicSmoothingSpline(x, y, p=0, w=w, bc_type=((1, 5.0), (1, 3.0)))
    np.testing.assert_allclose(s(x), par, atol=1e-7)

    s = CubicSmoothingSpline(x, y, p=0.5, bc_type=((1, 5.0), (2, 1.0)))
    np.testing.assert_allclose(s(x[0], nu=1), 5.0, atol=1e-10)
    np.testing.assert_allclose(s(x[-1], nu=2), 1.0, atol=1e-10)

    # derivative with respect to all inputs along a random direction, vs finite
    # differences
    def fun(t):
        xt, yt, wt = x + t[0] * dx, y + t[1] * dy, w + t[2] * dw
        bc = ((1, 5.0 + t[4]), (2, 1.0 - t[4]))
        s = CubicSmoothingSpline(xt, yt, p=0.3 + t[3], w=wt, bc_type=bc, check=False)
        return jnp.sum(s(xq) * cq) + jnp.sum(s(xq, nu=1) * cq[::-1])

    dx = rng.normal(size=x.size) * np.diff(x).min() / 4
    dy, dw = rng.normal(size=y.size), rng.normal(size=w.size) / 10
    cq = rng.normal(size=xq.size)
    t0 = jnp.zeros(5)
    grad = jax.jit(jax.grad(fun))(t0)
    # steps balance truncation error against roundoff from the closely spaced knots
    eps = [1e-2, 1e-2, 1e-2, 1e-4, 1e-2]
    fd = [
        (fun(t0.at[i].set(e)) - fun(t0.at[i].set(-e))) / (2 * e)
        for i, e in enumerate(eps)
    ]
    np.testing.assert_allclose(grad, fd, rtol=1e-3)


@pytest.mark.unit
@pytest.mark.skipif(not hasattr(jax, "shard_map"), reason="needs jax.shard_map")
@pytest.mark.parametrize("bc_type", ["not-a-knot", "natural", "clamped", "periodic"])
def test_approx_df_cubic2_inside_shard_map(bc_type):
    """approx_df works inside shard_map when f is an input and x is closed over.

    With jax >= 0.8, where f carries the manual mesh sharding inside shard_map, a
    lineax solve raised "Vector and operator structures do not match" here, since it
    compares the operator's sharding with the right-hand side's. Older jax doesn't
    attach that sharding, so there this test can't catch that regression.
    """
    x = jnp.log(jnp.geomspace(0.01, 1e3, 64))
    f = jnp.stack([jnp.sin(k * x) for k in (1.0, 2.0)])
    if bc_type == "periodic":
        f = f.at[:, -1].set(f[:, 0])
    # axis=0 rather than the default -1, which needs the fix in gh PR #211 for natural
    # and clamped boundary conditions
    df = jax.vmap(lambda v: approx_df(x, v, "cubic2", axis=0, bc_type=bc_type))
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("p",))
    spec = jax.sharding.PartitionSpec("p")
    sharded = jax.jit(jax.shard_map(df, mesh=mesh, in_specs=spec, out_specs=spec))
    np.testing.assert_allclose(sharded(f), df(f), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "bc_type", ["natural", "clamped", ((1, [0.5, -1.0]), "natural")]
)
def test_approx_df_cubic2_negative_axis(bc_type):
    """Boundary values get the shape of f without its interpolation axis."""
    x = jnp.linspace(0, 1, 10)
    f = jnp.stack([jnp.sin(x), jnp.cos(x)])
    np.testing.assert_allclose(
        approx_df(x, f, "cubic2", -1, bc_type=bc_type),
        approx_df(x, f.T, "cubic2", 0, bc_type=bc_type).T,
    )
    if isinstance(bc_type, str):
        # 1D f with the default axis=-1
        np.testing.assert_allclose(
            approx_df(x, f[0], "cubic2", bc_type=bc_type),
            approx_df(x, f.T, "cubic2", 0, bc_type=bc_type)[:, 0],
        )
    for axis in [2, -3]:
        with pytest.raises(ValueError):
            approx_df(x, f, "cubic2", axis, bc_type=bc_type)


@pytest.mark.unit
@pytest.mark.parametrize("derivative", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_interp1d_cubic_matches_hermite_spline(derivative, dtype):
    """interp1d with given fx is the cubic Hermite spline, for derivatives 0 to 3."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 3, 12))
    f = rng.normal(size=(12, 3, 2)).astype(dtype)
    fx = rng.normal(size=(12, 3, 2)).astype(dtype)
    xq = np.linspace(-0.5, 3.5, 50)
    out = interp1d(xq, x, f, "cubic", derivative, extrap=True, fx=fx)
    ref = CubicHermiteSpline(x, f, fx, axis=0)(xq, nu=derivative)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
    # a scalar query takes its own indexing path
    for q, r in [(xq[0], ref[0]), (xq[20], ref[20])]:
        out = interp1d(q, x, f, "cubic", derivative, extrap=True, fx=fx)
        np.testing.assert_allclose(out, r, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_interp1d_cubic_high_derivative_on_tiny_cell():
    """Derivatives above the third are zero even where 1/dx**4 overflows."""
    x = np.array([-1.0, 0.0, 1e-80, 1.0, 2.0])
    xq = np.array([5e-81, 1.5])
    for derivative in [4, 5]:
        out = interp1d(xq, x, x**2, "cubic", derivative)
        np.testing.assert_array_equal(out, 0)
        np.testing.assert_array_equal(interp1d(xq[0], x, x**2, "cubic", derivative), 0)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_interp1d_cubic_half_precision(dtype):
    """Half precision results stay close to the same rounded inputs in float64."""
    rng = np.random.default_rng(0)
    x = np.sort(np.concatenate([[0, 3], rng.uniform(0, 3, 38)]))
    f = np.sin(3 * x) + 0.01 * rng.normal(size=40)
    x, f, xq = (jnp.asarray(a, dtype) for a in (x, f, np.linspace(0.1, 2.9, 2000)))
    for derivative in [0, 1, 2]:
        out = interp1d(xq, x, f, "cubic", derivative)
        xq64, x64, f64 = (a.astype(float) for a in (xq, x, f))
        ref = interp1d(xq64, x64, f64, "cubic", derivative)
        assert out.dtype == dtype
        scale = np.max(np.abs(ref))
        # an empirical bound for this seed; with float16, some other seeds overflow
        np.testing.assert_allclose(out.astype(float), ref, atol=3e-2 * scale)


@pytest.mark.unit
def test_interp1d_cubic_mixed_dtypes():
    """The result has the dtype and precision of f and fx, not only of x."""
    rng = np.random.default_rng(1)
    x = np.sort(np.concatenate([[0, 3], rng.uniform(0, 3, 18)]))
    f, fx, g = np.sin(x), np.cos(x), rng.normal(size=20)
    xq = np.linspace(0, 3, 101)
    # real f with complex fx keeps the imaginary part of fx
    out = interp1d(xq, x, f, "cubic", 1, fx=fx + 1j * g)
    ref = CubicHermiteSpline(x, f, fx, axis=0)(xq, nu=1)
    ref = ref + 1j * CubicHermiteSpline(x, 0 * f, g, axis=0)(xq, nu=1)
    assert out.dtype == jnp.complex128
    np.testing.assert_allclose(out, ref, atol=1e-12)
    # float32 knots with float64 values keep float64 precision in the weights
    x32, xq32 = x.astype(np.float32), xq.astype(np.float32)
    out = interp1d(xq32, x32, f, "cubic", 1, fx=fx)
    ref = CubicHermiteSpline(x32.astype(float), f, fx, axis=0)(xq32.astype(float), nu=1)
    assert out.dtype == jnp.float64
    np.testing.assert_allclose(out, ref, atol=1e-9)


@pytest.mark.parametrize("derivative", [(0, 0, 0), (1, 0, 2), (0, 3, 1), (2, 2, 4)])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_interp2d_interp3d_cubic_are_tensor_products(derivative, dtype):
    """With separable data, cubic interp2d/3d are products of 1D Hermite splines."""
    rng = np.random.default_rng(0)
    grids = [np.sort(rng.uniform(0, 2, n)) for n in (7, 8, 6)]
    vals = [rng.normal(size=(len(g), 2)).astype(dtype) for g in grids]
    ders = [rng.normal(size=(len(g), 2)).astype(dtype) for g in grids]
    q = [np.linspace(-0.3, 2.3, 30) + 0.01 * k for k in range(3)]
    # 1D factors for each axis, with its derivative order
    h = [
        CubicHermiteSpline(g, v, d, axis=0)(qk, nu=n)
        for g, v, d, qk, n in zip(grids, vals, ders, q, derivative)
    ]

    def outer(fields):
        # the 2D/3D field built from one 1D factor per grid axis
        return functools.reduce(lambda a, b: a[..., None, :] * b, fields)

    for n, fun in [(2, interp2d), (3, interp3d)]:
        names = ["f", "fx", "fy", "fz", "fxy", "fxz", "fyz", "fxyz"]
        kw = {}
        for name in names:
            axes = "xyz"[:n]
            if any(c not in axes for c in name[1:]):
                continue
            kw[name] = outer(
                [ders[a] if c in name else vals[a] for a, c in enumerate(axes)]
            )
        f = kw.pop("f")
        out = fun(*q[:n], *grids[:n], f, "cubic", derivative[:n], extrap=True, **kw)
        ref = functools.reduce(lambda a, b: a * b, h[:n])
        np.testing.assert_allclose(out, ref, rtol=1e-11, atol=1e-11)


@pytest.mark.unit
def test_interp2d_interp3d_cubic_half_precision():
    """bfloat16 inputs are interpolated without rounding every Hermite term.

    The reference is the same rounded inputs, derivatives included, interpolated
    in float64.
    """
    rng = np.random.default_rng(0)
    grids = [np.sort(rng.uniform(0, 3, n)) for n in (12, 11, 10)]
    q = [rng.uniform(0.1, 2.9, 500) for _ in range(3)]
    for n, fun in [(2, interp2d), (3, interp3d)]:
        names = ("fx", "fy", "fz", "fxy", "fxz", "fyz", "fxyz")
        names = [m for m in names if set(m[1:]) <= set("xyz"[:n])]
        shape = tuple(len(g) for g in grids[:n])
        half = [jnp.asarray(a, jnp.bfloat16) for a in (*q[:n], *grids[:n])]
        fields = {m: jnp.asarray(rng.normal(size=shape), jnp.bfloat16) for m in names}
        f = jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
        for derivative in [(0,) * n, (1,) + (0,) * (n - 1), (2,) * n]:
            out = fun(*half, f, "cubic", derivative, extrap=True, **fields)
            ref = fun(
                *(a.astype(float) for a in half),
                f.astype(float),
                "cubic",
                derivative,
                extrap=True,
                **{m: v.astype(float) for m, v in fields.items()},
            )
            assert out.dtype == jnp.bfloat16
            scale = np.max(np.abs(ref))
            np.testing.assert_allclose(out.astype(float), ref, atol=6e-3 * scale)


@pytest.mark.unit
def test_interp2d_cubic_mixed_dtypes():
    """The result has the dtype and precision of f and its derivatives."""
    rng = np.random.default_rng(2)
    x = np.sort(np.concatenate([[0, 3], rng.uniform(0, 3, 8)]))
    y = np.sort(np.concatenate([[0, 3], rng.uniform(0, 3, 7)]))
    f, fx, fy, fxy, g = (rng.normal(size=(10, 9)) for _ in range(5))
    xq, yq = rng.uniform(0, 3, (2, 50))
    # real f with complex fx keeps the imaginary part of fx
    out = interp2d(xq, yq, x, y, f, "cubic", (1, 0), fx=fx + 1j * g, fy=fy, fxy=fxy)
    zero = 0 * f
    ref = interp2d(xq, yq, x, y, f, "cubic", (1, 0), fx=fx, fy=fy, fxy=fxy)
    ref = ref + 1j * interp2d(
        xq, yq, x, y, zero, "cubic", (1, 0), fx=g, fy=zero, fxy=zero
    )
    assert out.dtype == jnp.complex128
    np.testing.assert_allclose(out, ref, atol=1e-12)
    # float32 knots with float64 values keep float64 precision in the weights
    x32, y32, xq32, yq32 = (a.astype(np.float32) for a in (x, y, xq, yq))
    x64, y64, xq64, yq64 = (a.astype(float) for a in (x32, y32, xq32, yq32))
    out = interp2d(xq32, yq32, x32, y32, f, "cubic", (1, 1), fx=fx, fy=fy, fxy=fxy)
    ref = interp2d(xq64, yq64, x64, y64, f, "cubic", (1, 1), fx=fx, fy=fy, fxy=fxy)
    assert out.dtype == jnp.float64
    np.testing.assert_allclose(out, ref, atol=1e-9)
