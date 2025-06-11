"""Helpers for array types."""

from typing import Union

from jax import Array
from numpy.typing import ArrayLike

# jax.typing.ArrayLike and jaxtyping.ArrayLike don't include eg tuples,lists,iterables
# like np.ArrayLike. This combines all the usual array types
Arrayish = Union[Array, ArrayLike]
