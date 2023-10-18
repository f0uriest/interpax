"""interpax: interpolation and function approximation with JAX."""

from . import _version
from ._fourier import fft_interp1d, fft_interp2d
from ._spline import interp1d, interp2d, interp3d

__version__ = _version.get_versions()["version"]
