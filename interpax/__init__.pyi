from typing import Literal, Optional, Union

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
    ] = "cubic",
    axis: int = -1,
    **kwargs,
) -> Array: ...
def fft_interp1d(
    f: Array, n: int, sx: Optional[Array] = None, dx: float = ...
) -> Array: ...
def fft_interp2d(
    f: Array,
    n1: int,
    n2: int,
    sx: Optional[Array] = None,
    sy: Optional[Array] = None,
    dx: float = 1.0,
    dy: float = 1.0,
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
    ] = "cubic",
    derivative: int = 0,
    extrap: Union[bool, float, tuple] = False,
    period: Optional[float] = None,
    fx: Optional[Array] = None,
    axis: int = 0,
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
    ] = "cubic",
    derivative: int = 0,
    extrap: Union[bool, float, tuple] = False,
    period: Union[None, float, tuple] = None,
    fx: Optional[Array] = None,
    fy: Optional[Array] = None,
    fxy: Optional[Array] = None,
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
    ] = "cubic",
    derivative: int = 0,
    extrap: Union[bool, float, tuple] = False,
    period: Union[None, float, tuple] = None,
    fx: Optional[Array] = None,
    fy: Optional[Array] = None,
    fz: Optional[Array] = None,
    fxy: Optional[Array] = None,
    fxz: Optional[Array] = None,
    fyz: Optional[Array] = None,
    fxyz: Optional[Array] = None,
) -> Array: ...

class Interpolator1D:
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
        ] = "cubic",
        extrap: Union[bool, float, tuple] = False,
        period: Optional[float] = None,
        fx: Optional[Array] = None,
        axis: int = 0,
    ): ...
    def __call__(self, xq: Array, dx: int = 0): ...

class Interpolator2D:
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
        ] = "cubic",
        extrap: Union[bool, float, tuple] = False,
        period: Union[None, float, tuple] = None,
        axis: int = 0,
        fx: Optional[Array] = None,
        fy: Optional[Array] = None,
        fxy: Optional[Array] = None,
    ) -> None: ...
    def __call__(self, xq: Array, yq: Array, dx: int = 0, dy: int = 0): ...

class Interpolator3D:
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
        ] = "cubic",
        extrap: Union[bool, float, tuple] = False,
        period: Union[None, float, tuple] = None,
        fx: Optional[Array] = None,
        fy: Optional[Array] = None,
        fz: Optional[Array] = None,
        fxy: Optional[Array] = None,
        fxz: Optional[Array] = None,
        fyz: Optional[Array] = None,
        fxyz: Optional[Array] = None,
        axis: int = 0,
    ) -> None: ...
    def __call__(
        self,
        xq: Array,
        yq: Array,
        zq: Array,
        dx: int = 0,
        dy: int = 0,
        dz: int = 0,
    ): ...
