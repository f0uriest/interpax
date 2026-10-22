"""Setup/build/install script for interpax.

All static metadata lives in pyproject.toml. This shim exists only because
versioneer needs to inject its cmdclass, which cannot be declared declaratively.
"""

import versioneer
from setuptools import setup

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
