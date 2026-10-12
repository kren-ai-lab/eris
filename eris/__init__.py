"""Eris public API."""

from __future__ import annotations

from eris._version import __version__
from eris.api import embed, encode, pool
from eris.discovery import (
    get_embedder_info,
    get_representation_info,
    list_embedders,
    list_representations,
)
from eris.exceptions import RepresentationNotFoundError
from eris.export.pyg import to_pyg
from eris.export.torch import to_torch
from eris.pipelines.batch_encode import batch_embed, batch_encode
from eris.pipelines.pack import pack_artifacts as pack

__all__ = [
    "RepresentationNotFoundError",
    "__version__",
    "batch_embed",
    "batch_encode",
    "embed",
    "encode",
    "get_embedder_info",
    "get_representation_info",
    "list_embedders",
    "list_representations",
    "pack",
    "pool",
    "to_pyg",
    "to_torch",
]

# register built-in encoders
from eris import encoders as _encoders  # noqa: F401
