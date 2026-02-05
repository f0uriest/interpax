"""Module for RBF interpolation using JAX. Based on scipy implementation."""

import math
import warnings
from itertools import combinations_with_replacement
from typing import Any, Callable, Optional, Union, cast

import equinox as eqx
import jax
import jax.lax
import jax.numpy as jnp
import jaxkd as jk
from jax.scipy.linalg import solve
from jaxtyping import Array, Float, Int, Shaped
from typing_extensions import Literal

from .utils import asarray_inexact

__all__ = ["RBFInterpolator"]

# These RBFs are implemented
_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
}

# The shape parameter does not need to be specified when using these RBFs
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}

# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
}


def _monomial_powers(ndim: int, degree: int) -> Int[Array, " nmonos ndim"]:
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.
    """
    with jax.ensure_compile_time_eval():
        nmonos = math.comb(degree + ndim, ndim)
    out = jnp.zeros((nmonos, ndim), dtype=jnp.int32)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out = out.at[count, var].add(1)
            count += 1

    return out


def _rbf_kernel_direct(
    r2: Float[Array, "..."],
    kernel_func: Callable[[Float[Array, "..."]], Float[Array, "..."]],
) -> Float[Array, "..."]:
    """Evaluate the RBF kernel function with pre-scaled squared distances.

    Parameters
    ----------
    r2 : ndarray
        Squared distance between points (already scaled by epsilon)
    kernel_func : callable
        RBF kernel function for squared distances

    Returns
    -------
    ndarray
        Value of the RBF kernel
    """
    return kernel_func(r2)


def _build_system(
    y: Float[Array, " P N"],
    d: Shaped[Array, " P *d_shape"],
    smoothing: Float[Array, " P"],
    kernel_func: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    epsilon: float,
    powers: Int[Array, " R N"],
) -> tuple[
    Float[Array, " P+R P+R"],
    Shaped[Array, " P+R *d_shape"],
    Float[Array, " N"],
    Float[Array, " N"],
]:
    """Build the RBF interpolation system of equations.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel_func : callable
        RBF kernel function for squared distances.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    lhs : (P + R, P + R) float ndarray
        Left-hand side of the system.
    rhs : (P + R, S) float ndarray
        Right-hand side of the system.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.
    """
    P, N = y.shape
    R = powers.shape[0]

    # Shift and scale the polynomial domain to be between -1 and 1 (match SciPy)
    mins = jnp.min(y, axis=0)
    maxs = jnp.max(y, axis=0)
    shift = (maxs + mins) / 2
    scale = (maxs - mins) / 2
    # The scale may be zero if there is a single point or all the points have
    # the same value for some dimension. Avoid division by zero by replacing
    # zeros with ones.
    scale = jnp.where(scale == 0.0, 1.0, scale)

    # Apply epsilon scaling to coordinates (match SciPy order)
    yeps = y * epsilon
    yhat = (y - shift) / scale

    # Build the RBF matrix - use epsilon-scaled coordinates directly
    r2 = jnp.sum((yeps[:, None, :] - yeps[None, :, :]) ** 2, axis=2)
    K = _rbf_kernel_direct(r2, kernel_func)

    # Add smoothing to diagonal
    K = K + jnp.diag(smoothing)

    # Build the polynomial matrix using transformed coordinates
    if R > 0:
        poly_matrix = jnp.prod(yhat[:, None, :] ** powers[None, :, :], axis=2)
        lhs = jnp.block(
            [[K, poly_matrix], [poly_matrix.T, jnp.zeros((R, R), dtype=K.dtype)]]
        )
        rhs = jnp.block([[d], [jnp.zeros((R, d.shape[1]), dtype=d.dtype)]])
    else:
        lhs = K
        rhs = d

    return lhs, rhs, shift, scale


def _build_evaluation_coefficients(
    x: Float[Array, " Q N"],
    y: Float[Array, " P N"],
    kernel_func: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    epsilon: float,
    powers: Int[Array, " R N"],
    shift: Float[Array, " N"],
    scale: Float[Array, " N"],
) -> Float[Array, " Q P+R"]:
    """Build the coefficients for evaluating the RBF interpolant.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates.
    y : (P, N) float ndarray
        Data point coordinates.
    kernel_func : callable
        RBF kernel function for squared distances.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    Returns
    -------
    (Q, P + R) float ndarray
        Coefficients for evaluating the RBF interpolant.
    """
    Q, N = x.shape
    P, _ = y.shape
    R = powers.shape[0]

    # Apply epsilon scaling to coordinates (match SciPy order)
    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale

    # Build the RBF matrix using epsilon-scaled coordinates
    r2 = jnp.sum((xeps[:, None, :] - yeps[None, :, :]) ** 2, axis=2)
    K = _rbf_kernel_direct(r2, kernel_func)

    # Build the polynomial matrix using transformed coordinates
    if R > 0:
        poly_matrix = jnp.prod(xhat[:, None, :] ** powers[None, :, :], axis=2)
        return jnp.block([K, poly_matrix])
    else:
        return K


# Define individual kernel functions for JAX compatibility
def _linear_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Linear RBF kernel: -r."""
    return jnp.where(r2 > 0, -jnp.sqrt(r2), 0.0)


def _thin_plate_spline_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Thin plate spline RBF kernel: r^2 * log(r)."""
    safe_r2 = jnp.where(r2 > 0, r2, jnp.ones_like(r2))
    return jnp.where(r2 > 0, r2 * 0.5 * jnp.log(safe_r2), 0.0)


def _cubic_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Cubic RBF kernel: r^3."""
    return jnp.where(r2 > 0, r2 * jnp.sqrt(r2), 0.0)


def _quintic_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Quintic RBF kernel: -r^5."""
    return jnp.where(r2 > 0, -(r2 * r2 * jnp.sqrt(r2)), 0.0)


def _multiquadric_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Multiquadric RBF kernel: -sqrt(1 + r^2)."""
    return -jnp.sqrt(1 + r2)


def _inverse_multiquadric_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse multiquadric RBF kernel: 1/sqrt(1 + r^2)."""
    return 1 / jnp.sqrt(1 + r2)


def _inverse_quadratic_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Inverse quadratic RBF kernel: 1/(1 + r^2)."""
    return 1 / (1 + r2)


def _gaussian_kernel(r2: Float[Array, "..."]) -> Float[Array, "..."]:
    """Gaussian RBF kernel: exp(-r^2)."""
    return jnp.exp(-r2)


# Kernel function list
_KERNEL_FUNCTIONS = {
    "cubic": _cubic_kernel,
    "gaussian": _gaussian_kernel,
    "inverse_multiquadric": _inverse_multiquadric_kernel,
    "inverse_quadratic": _inverse_quadratic_kernel,
    "linear": _linear_kernel,
    "multiquadric": _multiquadric_kernel,
    "quintic": _quintic_kernel,
    "thin_plate_spline": _thin_plate_spline_kernel,
}


def _get_kernel(kernel: str) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    """Get the kernel function for a kernel name."""
    if kernel not in _AVAILABLE:
        raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")
    return _KERNEL_FUNCTIONS[kernel]


class RBFInterpolator(eqx.Module):
    """Radial basis function (RBF) interpolation in N dimensions.

    Parameters
    ----------
    y : (npoints, ndims) array_like
        2-D array of data point coordinates.
    d : (npoints, ...) array_like
        N-D array of data values at `y`. The length of `d` along the first
        axis must be equal to the length of `y`. Unlike some interpolators, the
        interpolation axis cannot be changed.
    neighbors : int, optional
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. All the data
        points are used by default.
    smoothing : float or (npoints, ) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree. Default is 0.
    kernel : str, optional
        Type of RBF. This should be one of

            - 'linear'               : ``-r``
            - 'thin_plate_spline'    : ``r**2 * log(r)``
            - 'cubic'                : ``r**3``
            - 'quintic'              : ``-r**5``
            - 'multiquadric'         : ``-sqrt(1 + r^2)``
            - 'inverse_multiquadric' : ``1/sqrt(1 + r^2)``
            - 'inverse_quadratic'    : ``1/(1 + r^2)``
            - 'gaussian'             : ``exp(-r^2)``

        Default is 'thin_plate_spline'.
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. If `kernel` is
        'linear', 'thin_plate_spline', 'cubic', or 'quintic', this defaults to
        1 and can be ignored because it has the same effect as scaling the
        smoothing parameter. Otherwise, this must be specified.
    degree : int, optional
        Degree of the added polynomial. For some RBFs the interpolant may not
        be well-posed if the polynomial degree is too small. Those RBFs and
        their corresponding minimum degrees are

            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2

        The default value is the minimum degree for `kernel` or 0 if there is
        no minimum degree. Set this to -1 for no added polynomial.
    """

    __hash__ = object.__hash__

    y: Float[Array, " P N"]
    d: Shaped[Array, " P *d_shape"]
    d_shape: tuple
    d_dtype: jnp.dtype = eqx.field(static=True)
    neighbors: Optional[int]
    smoothing: Float[Array, " P"]
    kernel: Literal[
        "cubic",
        "gaussian",
        "inverse_multiquadric",
        "inverse_quadratic",
        "linear",
        "multiquadric",
        "quintic",
        "thin_plate_spline",
    ] = eqx.field(static=True)
    kernel_func: Callable[[Float[Array, "..."]], Float[Array, "..."]] = eqx.field(
        static=True
    )
    epsilon: float
    powers: Int[Array, " R N"]
    _shift: Optional[Float[Array, " N"]]
    _scale: Optional[Float[Array, " N"]]
    _coeffs: Optional[Shaped[Array, " P+R *d_shape"]]
    _tree: Optional[Any]

    def __init__(
        self,
        y: Float[Array, " P N"],
        d: Shaped[Array, " P *d_shape"],
        neighbors: Optional[int] = None,
        smoothing: Union[float, Float[Array, " P"]] = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: Optional[float] = None,
        degree: Optional[int] = None,
    ):
        y = asarray_inexact(y)
        if y.ndim != 2:
            raise ValueError("`y` must be a 2-dimensional array.")

        ny, ndim = y.shape

        d = asarray_inexact(d)
        d_dtype = d.dtype
        if d.shape[0] != ny:
            raise ValueError(f"Expected the first axis of `d` to have length {ny}.")

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))
        # If `d` is complex, convert it to a float array with twice as many
        # columns. Otherwise, the LHS matrix would need to be converted to
        # complex and take up 2x more memory than necessary.
        if jnp.iscomplexobj(d):
            float_dtype = jnp.float32 if d_dtype == jnp.complex64 else jnp.float64
            d = d.view(float_dtype)

        if jnp.isscalar(smoothing):
            smoothing = jnp.full(ny, smoothing, dtype=y.dtype)
        else:
            smoothing = asarray_inexact(smoothing).astype(y.dtype)
            if smoothing.shape != (ny,):
                raise ValueError(
                    f"Expected `smoothing` to be a scalar or have shape ({ny},)."
                )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")
        kernel = cast(
            Literal[
                "cubic",
                "gaussian",
                "inverse_multiquadric",
                "inverse_quadratic",
                "linear",
                "multiquadric",
                "quintic",
                "thin_plate_spline",
            ],
            kernel,
        )

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    "`epsilon` must be specified if `kernel` is not one of "
                    f"{_SCALE_INVARIANT}."
                )
        else:
            epsilon = float(asarray_inexact(epsilon))

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif -1 < degree < min_degree:
                # Use standard warnings since inputs are static
                warning_msg = (
                    f"WARNING: `degree` should not be below {min_degree} except -1 "
                    f"when `kernel` is '{kernel}'. "
                    f"The interpolant may not be uniquely "
                    f"solvable, and the smoothing parameter may have an "
                    f"unintuitive effect."
                )
                warnings.warn(warning_msg, UserWarning)

        if neighbors is None:
            nobs = ny
        else:
            # Make sure the number of nearest neighbors used for interpolation
            # does not exceed the number of observations.
            neighbors = int(min(neighbors, ny))
            nobs = neighbors

        powers = _monomial_powers(ndim, degree)
        # The polynomial matrix must have full column rank in order for the
        # interpolant to be well-posed, which is not possible if there are
        # fewer observations than monomials.
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is {ndim}."
            )

        kernel_func = _get_kernel(kernel)

        if neighbors is None:
            lhs, rhs, shift, scale = _build_system(
                y, d, smoothing, kernel_func, epsilon, powers
            )
            coeffs = solve(lhs, rhs)
            self._shift = shift
            self._scale = scale
            self._coeffs = coeffs
            self._tree = None
        else:
            self._shift = None
            self._scale = None
            self._coeffs = None
            # Build the tree for nearest neighbor queries
            self._tree = jk.build_tree(y)

        self.y = y
        self.d = d
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.kernel_func = kernel_func
        self.epsilon = epsilon
        self.powers = powers

    def _chunk_evaluator(
        self,
        x: Float[Array, " Q N"],
        y: Float[Array, " P N"],
        shift: Float[Array, " N"],
        scale: Float[Array, " N"],
        coeffs: Shaped[Array, " P+R *d_shape"],
        memory_budget: int = 1000000,
    ) -> Shaped[Array, " Q *d_shape"]:
        """Evaluate the interpolation while controlling memory consumption.

        Parameters
        ----------
        x : (Q, N) float ndarray
            Array of points on which to evaluate
        y : (P, N) float ndarray
            Array of points on which we know function values
        shift : (N,) float ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs : (P + R, S) float ndarray
            Coefficients in front of basis functions
        memory_budget : int
            Total amount of memory (in units of sizeof(float)) we wish
            to devote for storing the array of coefficients for
            interpolated points. If we need more memory than that, we
            chunk the input.

        Returns
        -------
        (Q, S) float ndarray
            Interpolated array
        """
        nx, ndim = x.shape
        if self.neighbors is None:
            nnei = len(y)
        else:
            nnei = self.neighbors

        # in each chunk we consume the same space we already occupy
        chunksize = memory_budget // (self.powers.shape[0] + nnei) + 1

        if chunksize <= nx:
            pad = (-nx) % chunksize
            if pad:
                x_pad = jnp.pad(x, ((0, pad), (0, 0)), mode="edge")
            else:
                x_pad = x

            num_chunks = x_pad.shape[0] // chunksize
            x_chunks = x_pad.reshape((num_chunks, chunksize, ndim))

            def process_chunk(x_chunk):
                vec = _build_evaluation_coefficients(
                    x_chunk,
                    y,
                    self.kernel_func,
                    self.epsilon,
                    self.powers,
                    shift,
                    scale,
                )
                return jnp.dot(vec, coeffs)

            out_chunks = jax.lax.map(process_chunk, x_chunks)
            out = out_chunks.reshape((-1, self.d.shape[1]))[:nx]
        else:
            vec = _build_evaluation_coefficients(
                x, y, self.kernel_func, self.epsilon, self.powers, shift, scale
            )
            out = jnp.dot(vec, coeffs)

        return out

    def __call__(self, x: Float[Array, " Q N"]) -> Shaped[Array, " Q *d_shape"]:
        """Evaluate the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) array_like
            Evaluation point coordinates.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.
        """
        x = asarray_inexact(x)
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional array.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                f"Expected the second axis of `x` to have length {self.y.shape[1]}."
            )

        # Our memory budget for storing RBF coefficients is
        # based on how many floats in memory we already occupy
        # If this number is below 1e6 we just use 1e6
        # This memory budget is used to decide how we chunk
        # the inputs
        memory_budget = max(x.size + self.y.size + self.d.size, 1000000)

        if self.neighbors is None:
            if self._shift is None or self._scale is None or self._coeffs is None:
                raise ValueError("RBFInterpolator coefficients are not initialized.")
            out = self._chunk_evaluator(
                x,
                self.y,
                self._shift,
                self._scale,
                self._coeffs,
                memory_budget=memory_budget,
            )
        else:
            if self._tree is None:
                raise ValueError("RBFInterpolator neighbor tree is not initialized.")
            # Get the indices of the k nearest observation points to each
            # evaluation point.
            neighbors, _ = jk.query_neighbors(self._tree, x, k=self.neighbors)
            if self.neighbors == 1:
                # jaxkd may squeeze the output when k=1, ensure it's 2D
                neighbors = jnp.atleast_2d(neighbors).T

            out = jnp.empty((nx, self.d.shape[1]), dtype=self.d.dtype)

            # Process each evaluation point individually
            # This is simpler but less optimized than the scipy version
            def process_single_point(xi, neighbors_i):
                # Extract the neighborhood data
                ynbr = self.y[neighbors_i]
                dnbr = self.d[neighbors_i]
                snbr = self.smoothing[neighbors_i]

                # Build and solve the local system
                lhs, rhs, shift, scale = _build_system(
                    ynbr, dnbr, snbr, self.kernel_func, self.epsilon, self.powers
                )
                coeffs = solve(lhs, rhs)

                # Evaluate at the single query point (no chunking needed)
                xnbr = xi[None, :]  # Add batch dimension
                vec = _build_evaluation_coefficients(
                    xnbr,
                    ynbr,
                    self.kernel_func,
                    self.epsilon,
                    self.powers,
                    shift,
                    scale,
                )
                result = jnp.dot(vec, coeffs)

                return result[0]  # Extract the single result

            # Process points in chunks to stay within memory budget
            chunk_size = max(1, int(memory_budget / (self.neighbors * self.d.shape[1])))
            if chunk_size < nx:
                pad = (-nx) % chunk_size
                if pad:
                    x_pad = jnp.pad(x, ((0, pad), (0, 0)), mode="edge")
                    neighbors_pad = jnp.pad(neighbors, ((0, pad), (0, 0)), mode="edge")
                else:
                    x_pad = x
                    neighbors_pad = neighbors

                num_chunks = x_pad.shape[0] // chunk_size
                x_chunks = x_pad.reshape((num_chunks, chunk_size, ndim))
                neighbors_chunks = neighbors_pad.reshape(
                    (num_chunks, chunk_size, self.neighbors)
                )

                def process_chunk(chunk_data):
                    x_chunk, neighbors_chunk = chunk_data
                    return jax.vmap(process_single_point)(x_chunk, neighbors_chunk)

                out_chunks = jax.lax.map(process_chunk, (x_chunks, neighbors_chunks))
                out = out_chunks.reshape((-1, self.d.shape[1]))[:nx]
            else:
                out = jax.vmap(process_single_point)(x, neighbors)

        if jnp.issubdtype(self.d_dtype, jnp.complexfloating):
            out = out.astype(self.d.dtype).view(self.d_dtype)
        else:
            out = out.astype(self.d_dtype)
        out = out.reshape((nx,) + self.d_shape)
        return out
