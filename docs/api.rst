=================
API Documentation
=================

Interpolation of 1D, 2D, or 3D data
-----------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    interpax.Interpolator1D
    interpax.Interpolator2D
    interpax.Interpolator3D


``scipy.interpolate``-like classes
----------------------------------

These classes implement most of the functionality of the SciPy classes with the same names,
except where noted in the documentation.

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: class.rst

    interpax.Akima1DInterpolator
    interpax.CubicHermiteSpline
    interpax.CubicSpline
    interpax.PchipInterpolator
    interpax.PPoly


Functional interface for 1D, 2D, 3D interpolation
-------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    interpax.interp1d
    interpax.interp2d
    interpax.interp3d


Fourier interpolation of periodic functions in 1D and 2D
--------------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    interpax.fft_interp1d
    interpax.fft_interp2d
    interpax.ifft_interp1d
    interpax.ifft_interp2d


Approximating first derivatives for cubic splines
-------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    interpax.approx_df
