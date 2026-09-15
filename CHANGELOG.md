Changelog
=========

v0.3.15
-------
- ``CubicSpline`` now supports ``bc_type="periodic"``.
- Adds ``PPoly.solve`` and ``PPoly.roots``. Since output shapes must be static, these
take a ``size`` argument for the number of roots to return and a ``fill_value`` for
padding when fewer roots exist.
- ``fft_interp2d`` now raises an error if only one of ``sx``, ``sy`` is given.
- Importing interpax no longer initializes the JAX backend.

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.14...main


v0.3.14
-------
- Fix reverse mode AD for monotonic splines of a constant function

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.13...v0.3.14


v0.3.13
-------
- Fix evaluation of monotonic slopes in 3D
- Fixes for type promotion with mixed dtypes
- Dependency updates

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.12...v0.3.13


v0.3.12
-------
- Dependency updates

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.11...v0.3.12


v0.3.11
-------
- Improved support for complex dtypes
- Dependency updates

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.10...v0.3.11


v0.3.10
-------
- Dependency updates

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.9...v0.3.10


v0.3.9
------
- Package is now typed (``py.typed``), with cleaned up type hints
- ``bc_type`` for cubic splines is traceable where possible
- Adds testing for Python 3.13

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.8...v0.3.9


v0.3.8
------
- Adds ``AbstractInterpolator`` base class
- ``cubic2`` splines use ``lineax`` for the tridiagonal solve
- Fixes for duplicate ``x`` points and complex ``y``
- Fix output shape of ``PPoly.integrate`` and of extrapolated values for scalar ``x``
- Bump minimum JAX version

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.7...v0.3.8


v0.3.7
------
- Interpolator ``method`` is now a static field of the pytree
- Fix installation from sdist

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.6...v0.3.7


v0.3.6
------
- Drop support for Python 3.9
- Fix installation on Windows

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.5...v0.3.6


v0.3.5
------
- Dependency updates

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.4...v0.3.5


v0.3.4
------
- Support NumPy v2

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.3...v0.3.4


v0.3.3
------
- Reverts the change to periodic interpolation from v0.3.2

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.2...v0.3.3


v0.3.2
------
- Fix periodic transformation to avoid ``dx == 0``
- Ensure ``check`` argument is passed through correctly

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.1...v0.3.2


v0.3.1
------
- Dependency updates

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.3.0...v0.3.1


v0.3.0
------
- Adds a number of classes that replicate most of the functionality of the
corresponding classes from scipy.interpolate :
  - ``scipy.interpolate.PPoly`` -> ``interpax.PPoly``
  - ``scipy.interpolate.Akima1DInterpolator`` -> ``interpax.Akima1DInterpolator``
  - ``scipy.interpolate.CubicHermiteSpline`` -> ``interpax.CubicHermiteSpline``
  - ``scipy.interpolate.CubicSpline`` -> ``interpax.CubicSpline``
  - ``scipy.interpolate.PchipInterpolator`` -> ``interpax.PchipInterpolator``
- Method ``"akima"`` now available for ``Interpolator.{1D, 2D, 3D}`` and corresponding
functions.
- Method ``"monotonic"`` now works in 2D and 3D, where it will preserve monotonicity
with respect to each coordinate individually.

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.2.4...v0.3.0


v0.2.4
------
- Fixes for scalar valued query points
- Fixes for interpolating vector valued functions

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.2.3...v0.2.4


v0.2.3
------
- Add type annotations

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.2.2...v0.2.3


v0.2.2
------
- Add ``approx_df`` to public API

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.2.1...v0.2.2


v0.2.1
------
- More efficient nearest neighbor search
- Correct slopes for linear interpolation in 2d, 3d
- Fix for cubic2 splines in 2d and 3d
Forward and reverse mode AD now fully working and tested

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.2.0...v0.2.1


v0.2.0
-------
- Adds convenience classes for spline interpolation that cache the derivative calculation.

**Full Changelog**: https://github.com/f0uriest/interpax/compare/v0.1.0...v0.2.0


v0.1.0
------
Initial release
