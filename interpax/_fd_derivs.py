import jax
import jax.numpy as jnp
import lineax as lx
from jax import jit

from .utils import asarray_inexact, errorif


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
        - ``'cubic2'``: C2 cubic splines. Can also pass kwarg ``bc_type``, same as
          ``scipy.interpolate.CubicSpline``
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
    # noqa: D202

    # close over static args to deal with non-jittable kwargs
    def fun(x, f):
        return _approx_df(x, f, method, axis, **kwargs)

    return jit(fun)(x, f)


def _approx_df(x, f, method, axis, c=0, bc_type="not-a-knot"):
    if method == "cubic":
        out = _cubic1(x, f, axis)
    elif method == "cubic2":
        out = _cubic2(x, f, axis, bc_type=bc_type)
    elif method == "cardinal":
        out = _cardinal(x, f, axis, c=c)
    elif method == "catmull-rom":
        out = _cardinal(x, f, axis, c=0)
    elif method == "monotonic":
        out = _monotonic(x, f, axis, False)
    elif method == "monotonic-0":
        out = _monotonic(x, f, axis, True)
    elif method == "akima":
        out = _akima(x, f, axis)
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


def _validate_bc(bc_type, expected_deriv_shape, dtype):
    if isinstance(bc_type, str):
        errorif(bc_type == "periodic", NotImplementedError)
        bc_type = (bc_type, bc_type)

    else:
        errorif(
            len(bc_type) != 2,
            ValueError,
            "`bc_type` must contain 2 elements to specify start and end conditions.",
        )

        errorif(
            "periodic" in bc_type,
            ValueError,
            "'periodic' `bc_type` is defined for both "
            + "curve ends and cannot be used with other "
            + "boundary conditions.",
        )

    validated_bc = []
    for bc in bc_type:
        if isinstance(bc, str):
            errorif(bc_type == "periodic", NotImplementedError)
            if bc == "clamped":
                validated_bc.append((1, jnp.zeros(expected_deriv_shape)))
            elif bc == "natural":
                validated_bc.append((2, jnp.zeros(expected_deriv_shape)))
            elif bc in ["not-a-knot", "periodic"]:
                validated_bc.append(bc)
            else:
                raise ValueError(f"bc_type={bc} is not allowed.")
        else:
            try:
                deriv_order, deriv_value = bc
            except Exception as e:
                raise ValueError(
                    "A specified derivative value must be "
                    "given in the form (order, value)."
                ) from e

            if deriv_order not in [1, 2]:
                raise ValueError("The specified derivative order must " "be 1 or 2.")

            deriv_value = asarray_inexact(deriv_value)
            dtype = jnp.promote_types(dtype, deriv_value.dtype)
            if deriv_value.shape != expected_deriv_shape:
                raise ValueError(
                    "`deriv_value` shape {} is not the expected one {}.".format(
                        deriv_value.shape, expected_deriv_shape
                    )
                )
            validated_bc.append((deriv_order, deriv_value))
    return validated_bc, dtype


def _cubic2(x, f, axis, bc_type):
    f = jnp.moveaxis(f, axis, 0)
    bc, dtype = _validate_bc(bc_type, f.shape[1:], f.dtype)
    dx = jnp.diff(x)
    df = jnp.diff(f, axis=0)
    dxr = dx.reshape([dx.shape[0]] + [1] * (f.ndim - 1))
    dxi = jnp.where(dxr == 0, 0, 1 / jnp.where(dxr == 0, 1, dxr))
    df = dxi * df
    n = len(f)

    # If bc is 'not-a-knot' this change is just a convention.
    # If bc is 'periodic' then we already checked that y[0] == y[-1],
    # and the spline is just a constant, we handle this case in the
    # same way by setting the first derivatives to slope, which is 0.
    if n == 2:
        if bc[0] in ["not-a-knot", "periodic"]:
            bc[0] = (1, df[0])
        if bc[1] in ["not-a-knot", "periodic"]:
            bc[1] = (1, df[0])

    # This is a special case, when both conditions are 'not-a-knot'
    # and n == 3. In this case 'not-a-knot' can't be handled regularly
    # as the both conditions are identical. We handle this case by
    # constructing a parabola passing through given points.
    if n == 3 and bc[0] == "not-a-knot" and bc[1] == "not-a-knot":
        A = jnp.zeros((3, 3))  # This is a standard matrix.
        b = jnp.empty((3,) + f.shape[1:], dtype=dtype)

        A = A.at[0, 0].set(1)
        A = A.at[0, 1].set(1)
        A = A.at[1, 0].set(dx[1])
        A = A.at[1, 1].set(2 * (dx[0] + dx[1]))
        A = A.at[1, 2].set(dx[0])
        A = A.at[2, 1].set(1)
        A = A.at[2, 2].set(1)

        b = b.at[0].set(2 * df[0])
        b = b.at[1].set(3 * (dxr[0] * df[1] + dxr[1] * df[0]))
        b = b.at[2].set(2 * df[1])

        solve = lambda b: jnp.linalg.solve(A, b)
        fx = jnp.vectorize(solve, signature="(n)->(n)")(b.T).T
        fx = jnp.moveaxis(fx, 0, axis)

    else:

        # Find derivative values at each x[i] by solving a tridiagonal
        # system.
        diag = jnp.zeros(n, dtype=x.dtype)
        diag = diag.at[1:-1].set(2 * (dx[:-1] + dx[1:]))
        upper_diag = jnp.zeros(n - 1, dtype=x.dtype)
        upper_diag = upper_diag.at[1:].set(dx[:-1])
        lower_diag = jnp.zeros(n - 1, dtype=x.dtype)
        lower_diag = lower_diag.at[:-1].set(dx[1:])
        b = jnp.zeros((n,) + f.shape[1:], dtype=dtype)
        b = b.at[1:-1].set(3 * (dxr[1:] * df[:-1] + dxr[:-1] * df[1:]))

        bc_start, bc_end = bc

        if bc_start == "not-a-knot":
            d = x[2] - x[0]
            diag = diag.at[0].set(dx[1])
            upper_diag = upper_diag.at[0].set(d)
            b = b.at[0].set(
                ((dxr[0] + 2 * d) * dxr[1] * df[0] + dxr[0] ** 2 * df[1]) / d
            )
        elif bc_start[0] == 1:
            diag = diag.at[0].set(1)
            upper_diag = upper_diag.at[0].set(0)
            b = b.at[0].set(bc_start[1])
        elif bc_start[0] == 2:
            diag = diag.at[0].set(2 * dx[0])
            upper_diag = upper_diag.at[0].set(dx[0])
            b = b.at[0].set(-0.5 * bc_start[1] * dx[0] ** 2 + 3 * (f[1] - f[0]))

        if bc_end == "not-a-knot":
            d = x[-1] - x[-3]
            diag = diag.at[-1].set(dx[-2])
            lower_diag = lower_diag.at[-1].set(d)
            b = b.at[-1].set(
                (dxr[-1] ** 2 * df[-2] + (2 * d + dxr[-1]) * dxr[-2] * df[-1]) / d
            )
        elif bc_end[0] == 1:
            diag = diag.at[-1].set(1)
            lower_diag = lower_diag.at[-1].set(0)
            b = b.at[-1].set(bc_end[1])
        elif bc_end[0] == 2:
            diag = diag.at[-1].set(2 * dx[-1])
            lower_diag = lower_diag.at[-1].set(dx[-1])
            b = b.at[-1].set(0.5 * bc_end[1] * dx[-1] ** 2 + 3 * (f[-1] - f[-2]))

        # this is needed to avoid singular matrix when there are duplicate x coords
        mask = diag == 0
        diag = jnp.where(mask, 1, diag)
        lower_diag = jnp.where(mask[1:], 0, lower_diag)
        upper_diag = jnp.where(mask[:-1], 0, upper_diag)
        b = jnp.where(mask, 0, b.T).T

        # see https://github.com/patrick-kidger/lineax/issues/148
        dtype = jnp.result_type(diag, lower_diag, upper_diag, b)
        diag = diag.astype(dtype)
        lower_diag = lower_diag.astype(dtype)
        upper_diag = upper_diag.astype(dtype)
        b = b.astype(dtype)

        A = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)

        solve = lambda b: lx.linear_solve(A, b, lx.Tridiagonal()).value
        fx = jnp.vectorize(solve, signature="(n)->(n)")(b.T).T
        fx = jnp.moveaxis(fx, 0, axis)
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
