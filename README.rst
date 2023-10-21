
########
interpax
########
|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |Codecov|

interpax is a library for interpolation and function approximation using JAX.

Includes methods for nearest neighbor, linear, and several cubic interpolation schemes
in 1d, 2d, and 3d, as well as Fourier interpolation for periodic functions in
1d and 2d.

Coming soon:
- Spline interpolation for rectilinear grids in N-dimensions
- RBF interpolation for unstructured data in N-dimensions
- Smoothing splines for noisy data


Installation
============

interpax is installable with `pip`:

.. code-block:: console

    pip install interpax



Usage
=====

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np
    from interpax import interp1d

    xp = jnp.linspace(0, 2 * np.pi, 100)
    xq = jnp.linspace(0, 2 * np.pi, 10000)
    f = lambda x: jnp.sin(x)
    fp = f(xp)

    fq = interp1d(xq, xp, fp, method="cubic")
    np.testing.assert_allclose(fq, f(xq), rtol=1e-6, atol=1e-5)


For full details of various options see the `API documentation <https://interpax.readthedocs.io/en/latest/api.html>`__


.. |License| image:: https://img.shields.io/github/license/f0uriest/interpax?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/f0uriest/interpax/blob/master/LICENSE
    :alt: License

.. |DOI| image:: https://zenodo.org/badge/706703896.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10028967
    :alt: DOI

.. |Docs| image:: https://img.shields.io/readthedocs/interpax?logo=Read-the-Docs
    :target: https://interpax.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |UnitTests| image:: https://github.com/f0uriest/interpax/actions/workflows/unittest.yml/badge.svg
    :target: https://github.com/f0uriest/interpax/actions/workflows/unittest.yml
    :alt: UnitTests

.. |Codecov| image:: https://codecov.io/github/f0uriest/interpax/graph/badge.svg?token=MB11I7WE3I
    :target: https://codecov.io/github/f0uriest/interpax
    :alt: Coverage

.. |Issues| image:: https://img.shields.io/github/issues/f0uriest/interpax
    :target: https://github.com/f0uriest/interpax/issues
    :alt: GitHub issues

.. |Pypi| image:: https://img.shields.io/pypi/v/interpax
    :target: https://pypi.org/project/interpax/
    :alt: Pypi
