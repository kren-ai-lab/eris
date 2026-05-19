from __future__ import annotations

# register built-in encoders
from eris import encoders as _encoders  # noqa: F401
from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("eris")
except PackageNotFoundError:  # pragma: no cover
    # Happens when running from source tree without an installed package
    __version__ = "0.0.0"
