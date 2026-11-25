"""Matrices/tensors for getting spline coefficients from derivatives."""

import numpy as np

# fmt: off
A_CUBIC = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [-3, 3, -2, -1],
    [2, -2,  1,  1],
])
