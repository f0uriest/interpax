from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=("method", "axis"))
def approx_df(
    x: jax.Array, f: jax.Array, method: str = "cubic", axis: int = -1, **kwargs
):
    """Approximates first derivatives using cubic spline interpolation.

    Parameters
    ----------
    x : ndarray, shape(Nx,)
        coordinates of known function values ("knots")
    f : ndarray
        Known function values. Should have length ``Nx`` along axis=axis
    method : str
        method of approximation

        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, can also pass
          keyword parameter ``c`` in float[0,1] to specify tension
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as ``'monotonic'`` but with 0 first derivatives at
          both endpoints
        - ``'akima'``: C1 cubic splines that appear smooth and natural

    axis : int
        Axis along which f is varying.

    Returns
    -------
    df : ndarray, shape(f.shape)
        First derivative of f with respect to x.

    """
    if method == "cubic":
        out = _cubic1(x, f, axis, **kwargs)
    elif method == "cubic2":
        out = _cubic2(x, f, axis)
    elif method == "cardinal":
        out = _cardinal(x, f, axis, **kwargs)
    elif method == "catmull-rom":
        out = _cardinal(x, f, axis, **kwargs)
    elif method == "monotonic":
        out = _monotonic(x, f, axis, False, **kwargs)
    elif method == "monotonic-0":
        out = _monotonic(x, f, axis, True, **kwargs)
    elif method == "akima":
        out = _akima(x, f, axis, **kwargs)
    elif method in ("nearest", "linear"):
        out = jnp.zeros_like(f)
    else:
        raise ValueError(f"got unknown method {method}")
    return out


def _cubic1(x, f, axis):
    dx = jnp.diff(x)
    df = jnp.diff(f, axis=axis)
    dxi = jnp.where(dx == 0, 0, 1 / dx)
    if df.ndim > dxi.ndim:
        dxi = jnp.expand_dims(dxi, tuple(range(1, df.ndim)))
        dxi = jnp.moveaxis(dxi, 0, axis)
    df = dxi * df
    fx = jnp.concatenate(
        [
            jnp.take(df, jnp.array([0]), axis, mode="wrap"),
            1
            / 2
            * (
                jnp.take(df, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                + jnp.take(df, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
            ),
            jnp.take(df, jnp.array([-1]), axis, mode="wrap"),
        ],
        axis=axis,
    )
    return fx


def _cubic2(x, f, axis):
    dx = jnp.diff(x)
    df = jnp.diff(f, axis=axis)
    if df.ndim > dx.ndim:
        dx = jnp.expand_dims(dx, tuple(range(1, df.ndim)))
        dx = jnp.moveaxis(dx, 0, axis)
    dxi = jnp.where(dx == 0, 0, 1 / dx)
    df = dxi * df

    one = jnp.array([1.0])
    dxflat = dx.flatten()
    diag = jnp.concatenate([one, 2 * (dxflat[:-1] + dxflat[1:]), one])
    upper_diag = jnp.concatenate([one, dxflat[:-1]])
    lower_diag = jnp.concatenate([dxflat[1:], one])

    A = jnp.diag(diag) + jnp.diag(upper_diag, k=1) + jnp.diag(lower_diag, k=-1)
    b = jnp.concatenate(
        [
            2 * jnp.take(df, jnp.array([0]), axis, mode="wrap"),
            3
            * (
                jnp.take(dx, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                * jnp.take(df, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                + jnp.take(dx, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                * jnp.take(df, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
            ),
            2 * jnp.take(df, jnp.array([-1]), axis, mode="wrap"),
        ],
        axis=axis,
    )
    ba = jnp.moveaxis(b, axis, 0)
    br = ba.reshape((b.shape[axis], -1))
    solve = lambda b: jnp.linalg.solve(A, b)
    fx = jnp.vectorize(solve, signature="(n)->(n)")(br.T).T
    fx = jnp.moveaxis(fx.reshape(ba.shape), 0, axis)
    return fx


def _cardinal(x, f, axis, c=0):
    dx = x[2:] - x[:-2]
    df = jnp.take(f, jnp.arange(2, f.shape[axis]), axis, mode="wrap") - jnp.take(
        f, jnp.arange(0, f.shape[axis] - 2), axis, mode="wrap"
    )
    dxi = jnp.where(dx == 0, 0, 1 / dx)
    if df.ndim > dxi.ndim:
        dxi = jnp.expand_dims(dxi, tuple(range(1, df.ndim)))
        dxi = jnp.moveaxis(dxi, 0, axis)
    df = dxi * df
    fx0 = jnp.take(f, jnp.array([1]), axis, mode="wrap") - jnp.take(
        f, jnp.array([0]), axis, mode="wrap"
    )
    fx0 *= jnp.where(x[0] == x[1], 0, 1 / (x[1] - x[0]))
    fx1 = jnp.take(f, jnp.array([-1]), axis, mode="wrap") - jnp.take(
        f, jnp.array([-2]), axis, mode="wrap"
    )
    fx1 *= jnp.where(x[-1] == x[-2], 0, 1 / (x[-1] - x[-2]))

    fx = (1 - c) * jnp.concatenate([fx0, df, fx1], axis=axis)
    return fx


def _monotonic(x, f, axis, zero_slope):
    f = jnp.moveaxis(f, axis, 0)
    fshp = f.shape
    if f.ndim == 1:
        # So that _edge_case doesn't end up assigning to scalars
        x = x[:, None]
        f = f[:, None]
    hk = x[1:] - x[:-1]
    df = jnp.diff(f, axis=0)
    hki = jnp.where(hk == 0, 0, 1 / hk)
    if df.ndim > hki.ndim:
        hki = jnp.expand_dims(hki, tuple(range(1, df.ndim)))

    mk = hki * df

    smk = jnp.sign(mk)
    condition = (smk[1:, :] != smk[:-1, :]) | (mk[1:, :] == 0) | (mk[:-1, :] == 0)

    w1 = 2 * hk[1:] + hk[:-1]
    w2 = hk[1:] + 2 * hk[:-1]

    if df.ndim > w1.ndim:
        w1 = jnp.expand_dims(w1, tuple(range(1, df.ndim)))
        w2 = jnp.expand_dims(w2, tuple(range(1, df.ndim)))

    whmean = (w1 / mk[:-1, :] + w2 / mk[1:, :]) / (w1 + w2)

    dk = jnp.where(condition, 0, 1.0 / whmean)

    if zero_slope:
        d0 = jnp.zeros((1, dk.shape[1]))
        d1 = jnp.zeros((1, dk.shape[1]))

    else:
        # special case endpoints, as suggested in
        # Cleve Moler, Numerical Computing with MATLAB, Chap 3.6 (pchiptx.m)
        def _edge_case(h0, h1, m0, m1):
            # one-sided three-point estimate for the derivative
            d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

            # try to preserve shape
            mask = jnp.sign(d) != jnp.sign(m0)
            mask2 = (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > 3.0 * jnp.abs(m0))
            mmm = (~mask) & mask2

            d = jnp.where(mask, 0.0, d)
            d = jnp.where(mmm, 3.0 * m0, d)
            return d

        hk = 1 / hki
        d0 = _edge_case(hk[0, :], hk[1, :], mk[0, :], mk[1, :])[None]
        d1 = _edge_case(hk[-1, :], hk[-2, :], mk[-1, :], mk[-2, :])[None]

    dk = jnp.concatenate([d0, dk, d1])
    dk = dk.reshape(fshp)
    return jnp.moveaxis(dk, 0, axis)


def _akima(x, f, axis):
    # Original implementation in MATLAB by N. Shamsundar (BSD licensed), see
    # https://www.mathworks.com/matlabcentral/fileexchange/1814-akima-interpolation
    dx = jnp.diff(x)
    f = jnp.moveaxis(f, axis, 0)
    # determine slopes between breakpoints
    m = jnp.empty((x.size + 3,) + f.shape[1:])
    dx = dx[(slice(None),) + (None,) * (f.ndim - 1)]
    mask = dx == 0
    dx = jnp.where(mask, 1, dx)
    dxi = jnp.where(mask, 0.0, 1 / dx)
    m = m.at[2:-2].set(jnp.diff(f, axis=0) * dxi)

    # add two additional points on the left ...
    m = m.at[1].set(2.0 * m[2] - m[3])
    m = m.at[0].set(2.0 * m[1] - m[2])
    # ... and on the right
    m = m.at[-2].set(2.0 * m[-3] - m[-4])
    m = m.at[-1].set(2.0 * m[-2] - m[-3])

    # df = derivative of f at x
    # df = (|m4 - m3| * m2 + |m2 - m1| * m3) / (|m4 - m3| + |m2 - m1|)
    # if m1 == m2 != m3 == m4, the slope at the breakpoint is not
    # defined. Use instead 1/2(m2 + m3)
    dm = jnp.abs(jnp.diff(m, axis=0))
    m2 = m[1:-2]
    m3 = m[2:-1]
    m4m3 = dm[2:]  # |m4 - m3|
    m2m1 = dm[:-2]  # |m2 - m1|
    f12 = m4m3 + m2m1
    mask = f12 > 1e-9 * jnp.max(f12, initial=-jnp.inf)
    df = (m4m3 * m2 + m2m1 * m3) / jnp.where(mask, f12, 1.0)
    df = jnp.where(mask, df, 0.5 * (m[3:] + m[:-3]))
    return jnp.moveaxis(df, 0, axis)
