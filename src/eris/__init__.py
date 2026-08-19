from __future__ import annotations

from importlib.metadata import version as distribution_version

__all__ = ["__version__"]

__version__ = distribution_version("eris")

# register built-in encoders
from eris import encoders as _encoders  # noqa: F401
