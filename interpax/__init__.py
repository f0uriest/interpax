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
    interp1d,
    interp2d,
    interp3d,
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
    "interp1d",
    "interp2d",
    "interp3d",
]

__version__ = _version.get_versions()["version"]


def _deprecated_submodules(names):
    """Make each ``interpax.<name>`` a deprecated alias of ``interpax._<name>``."""
    import importlib
    import sys
    import types
    import warnings

    def warn(name):
        warnings.warn(
            f"`{__name__}.{name}` is not part of the public API and importing it is "
            f"deprecated. Import from `{__name__}` instead; the old module path will "
            "be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )

    for name in names:
        private = importlib.import_module(f"{__name__}._{name}")
        alias = types.ModuleType(f"{__name__}.{name}")

        def alias_getattr(attr, name=name, private=private):
            # Tools that scan sys.modules, such as inspect.getmodule, probe dunder
            # attributes of every module, so those must not warn or be forwarded.
            if attr[:2] == attr[-2:] == "__":
                raise AttributeError(attr)
            warn(name)
            return getattr(private, attr)

        setattr(alias, "__all__", [k for k in vars(private) if not k.startswith("_")])
        setattr(alias, "__getattr__", alias_getattr)
        # Since the alias is already in sys.modules, importing it never binds it as an
        # attribute of the package, so it cannot shadow a public attribute.
        sys.modules[alias.__name__] = alias

    def package_getattr(name):
        if name not in names:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        warn(name)
        return sys.modules[f"{__name__}.{name}"]

    return package_getattr


# Assigned through globals() so that static type checkers do not treat arbitrary
# attributes of the package as valid.
globals()["__getattr__"] = _deprecated_submodules(["utils"])
