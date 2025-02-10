from typing import Literal, Optional, Union

import equinox as eqx
from jax import Array

def approx_df(
    x: Array,
    f: Array,
    method: Literal[
        "nearest",
        "linear",
        "cubic",
        "cubic2",
        "cardinal",
        "catmull-rom",
        "monotonic",
        "monotonic-0",
        "akima",
    ] = ...,
    axis: int = ...,
    **kwargs,
) -> Array: ...
def fft_interp1d(
    f: Array, n: int, sx: Optional[Array] = ..., dx: float = ...
) -> Array: ...
def fft_interp2d(
    f: Array,
    n1: int,
    n2: int,
    sx: Optional[Array] = ...,
    sy: Optional[Array] = ...,
    dx: float = ...,
    dy: float = ...,
) -> Array: ...
def interp1d(
    xq: Array,
    x: Array,
    f: Array,
    method: Literal[
        "nearest",
        "linear",
        "cubic",
        "cubic2",
        "cardinal",
        "catmull-rom",
        "akima",
        "monotonic",
        "monotonic-0",
    ] = ...,
    derivative: int = ...,
    extrap: Union[bool, float, tuple] = ...,
    period: Optional[float] = ...,
    fx: Optional[Array] = ...,
    axis: int = ...,
) -> Array: ...
def interp2d(
    xq: Array,
    yq: Array,
    x: Array,
    y: Array,
    f: Array,
    method: Literal[
        "nearest",
        "linear",
        "cubic",
        "cubic2",
        "cardinal",
        "catmull-rom",
        "akima",
        "monotonic",
        "monotonic-0",
    ] = ...,
    derivative: int = ...,
    extrap: Union[bool, float, tuple] = ...,
    period: Union[None, float, tuple] = ...,
    fx: Optional[Array] = ...,
    fy: Optional[Array] = ...,
    fxy: Optional[Array] = ...,
) -> Array: ...
def interp3d(
    xq: Array,
    yq: Array,
    zq: Array,
    x: Array,
    y: Array,
    z: Array,
    f: Array,
    method: Literal[
        "nearest",
        "linear",
        "cubic",
        "cubic2",
        "cardinal",
        "catmull-rom",
        "akima",
        "monotonic",
        "monotonic-0",
    ] = ...,
    derivative: int = ...,
    extrap: Union[bool, float, tuple] = ...,
    period: Union[None, float, tuple] = ...,
    fx: Optional[Array] = ...,
    fy: Optional[Array] = ...,
    fz: Optional[Array] = ...,
    fxy: Optional[Array] = ...,
    fxz: Optional[Array] = ...,
    fyz: Optional[Array] = ...,
    fxyz: Optional[Array] = ...,
) -> Array: ...

class Interpolator1D(eqx.Module):
    x: Array
    f: Array
    derivs: dict
    method: str
    extrap: Union[bool, float, tuple]
    period: Optional[float]
    axis: int

    def __init__(
        self,
        x: Array,
        f: Array,
        method: Literal[
            "nearest",
            "linear",
            "cubic",
            "cubic2",
            "cardinal",
            "catmull-rom",
            "akima",
            "monotonic",
            "monotonic-0",
        ] = ...,
        extrap: Union[bool, float, tuple] = ...,
        period: Optional[float] = ...,
        fx: Optional[Array] = ...,
        axis: int = ...,
    ): ...
    def __call__(self, xq: Array, dx: int = 0): ...

class Interpolator2D(eqx.Module):
    x: Array
    y: Array
    f: Array
    derivs: dict
    method: str
    extrap: Union[bool, float, tuple]
    period: Union[None, float, tuple]
    axis: int

    def __init__(
        self,
        x: Array,
        y: Array,
        f: Array,
        method: Literal[
            "nearest",
            "linear",
            "cubic",
            "cubic2",
            "cardinal",
            "catmull-rom",
            "akima",
            "monotonic",
            "monotonic-0",
        ] = ...,
        extrap: Union[bool, float, tuple] = ...,
        period: Union[None, float, tuple] = ...,
        axis: int = ...,
        fx: Optional[Array] = ...,
        fy: Optional[Array] = ...,
        fxy: Optional[Array] = ...,
    ) -> None: ...
    def __call__(self, xq: Array, yq: Array, dx: int = 0, dy: int = 0): ...

class Interpolator3D(eqx.Module):
    x: Array
    y: Array
    z: Array
    f: Array
    derivs: dict
    method: str
    extrap: Union[bool, float, tuple]
    period: Union[None, float, tuple]
    axis: int

    def __init__(
        self,
        x: Array,
        y: Array,
        z: Array,
        f: Array,
        method: Literal[
            "nearest",
            "linear",
            "cubic",
            "cubic2",
            "cardinal",
            "catmull-rom",
            "akima",
            "monotonic",
            "monotonic-0",
        ] = ...,
        extrap: Union[bool, float, tuple] = ...,
        period: Union[None, float, tuple] = ...,
        fx: Optional[Array] = ...,
        fy: Optional[Array] = ...,
        fz: Optional[Array] = ...,
        fxy: Optional[Array] = ...,
        fxz: Optional[Array] = ...,
        fyz: Optional[Array] = ...,
        fxyz: Optional[Array] = ...,
        axis: int = ...,
    ) -> None: ...
    def __call__(
        self,
        xq: Array,
        yq: Array,
        zq: Array,
        dx: int = ...,
        dy: int = ...,
        dz: int = ...,
    ): ...

class PPoly(eqx.Module):
    def __init__(
        self,
        c: Array,
        x: Array,
        extrapolate: Union[None, bool, str] = ...,
        axis: int = ...,
        check: bool = ...,
    ) -> None: ...
    @property
    def c(self) -> Array: ...
    @property
    def x(self) -> Array: ...
    @property
    def extrapolate(self) -> Union[bool, str]: ...
    @property
    def axis(self) -> int: ...
    @classmethod
    def construct_fast(
        cls,
        c: Array,
        x: Array,
        extrapolate: Union[None, bool, Literal["periodic"]] = ...,
        axis: int = ...,
    ) -> "PPoly": ...
    def __call__(
        self,
        x: Array,
        nu: int = ...,
        extrapolate: Union[None, bool, Literal["periodic"]] = ...,
    ) -> Array: ...
    def derivative(self, nu: int = ...) -> "PPoly": ...
    def antiderivative(self, nu: int = ...) -> "PPoly": ...
    def integrate(
        self,
        a: float,
        b: float,
        extrapolate: Union[None, bool, Literal["periodic"]] = ...,
    ) -> Array: ...
    def solve(self, y=..., discontinuity=..., extrapolate=...): ...
    def roots(self, discontinuity=..., extrapolate=...): ...
    def extend(self, c, x, right=...): ...
    @classmethod
    def from_spline(cls, tck, extrapolate=...): ...
    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=...): ...

class CubicHermiteSpline(PPoly):
    def __init__(
        self,
        x: Array,
        y: Array,
        dydx: Array,
        axis: int = ...,
        extrapolate: Union[None, bool, Literal["periodic"]] = ...,
        check: bool = ...,
    ) -> None: ...

class PchipInterpolator(CubicHermiteSpline):
    def __init__(
        self,
        x: Array,
        y: Array,
        axis: int = ...,
        extrapolate: Union[None, bool, Literal["periodic"]] = ...,
        check: bool = ...,
    ) -> None: ...

class Akima1DInterpolator(CubicHermiteSpline):
    def __init__(
        self,
        x: Array,
        y: Array,
        axis: int = ...,
        extrapolate: Union[None, bool, Literal["periodic"]] = ...,
        check: bool = ...,
    ) -> None: ...

class CubicSpline(CubicHermiteSpline):
    def __init__(
        self,
        x: Array,
        y: Array,
        axis: int = ...,
        bc_type: Union[str, tuple] = ...,
        extrapolate: Union[None, bool, Literal["periodic"]] = ...,
        check: bool = ...,
    ) -> None: ...
