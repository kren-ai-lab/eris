from __future__ import annotations

from importlib.metadata import version as distribution_version

from eris.api import embed, encode
from eris.discovery import (
    get_embedder_info,
    get_representation_info,
    list_embedders,
    list_representations,
)
from eris.exceptions import RepresentationNotFoundError

__all__ = [
    "RepresentationNotFoundError",
    "__version__",
    "encode",
    "embed",
    "get_embedder_info",
    "get_representation_info",
    "list_embedders",
    "list_representations",
]

__version__ = distribution_version("eris")

# register built-in encoders
from eris import encoders as _encoders  # noqa: F401
