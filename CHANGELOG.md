Changelog
=========

- [Improves FFT interpolation](https://github.com/f0uriest/interpax/pull/116)
  - The real FFT is now used where possible.
  - Double the width of the Fourier spectrum is now preserved when interpolating to a less dense grid, at no additional cost.
  - In the 2D upsampling case, the second transform is now padded only after computing the first transform. In the 2D downsampling case, the second transform is now truncated prior to computing the first transform. This reduces the size of the problem, so the computation is less expensive.
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
