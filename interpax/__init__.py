"""interpax: interpolation and function approximation with JAX."""

from . import _version
from ._fd_derivs import approx_df
from ._fourier import fft_interp1d, fft_interp2d
from ._ppoly import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
    PPoly,
)
from ._spline import (
    AbstractInterpolator,
    Interpolator1D,
    Interpolator2D,
    Interpolator3D,
    InterpolatorNd,
    interp1d,
    interp2d,
    interp3d,
    interpNd,
)

__all__ = [
    "approx_df",
    "fft_interp1d",
    "fft_interp2d",
    "Akima1DInterpolator",
    "CubicHermiteSpline",
    "CubicSpline",
    "PchipInterpolator",
    "PPoly",
    "AbstractInterpolator",
    "Interpolator1D",
    "Interpolator2D",
    "Interpolator3D",
    "InterpolatorNd",
    "interp1d",
    "interp2d",
    "interp3d",
    "interpNd",
]

__version__ = _version.get_versions()["version"]
