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

__version__ = distribution_version("eris")

# M7 workflow entry points.  These imports expose the existing canonical boundaries;
# implementation remains in the pipelines/export modules.
from eris.export.pyg import save_artifact_to_pyg, to_pyg  # noqa: F401
from eris.export.torch import save_artifact_to_torch, to_torch  # noqa: F401
from eris.pipelines.batch_encode import batch_embed, batch_encode  # noqa: F401
from eris.pipelines.batch_export import batch_export  # noqa: F401
from eris.pipelines.pack import pack_artifacts  # noqa: F401

export_torch = save_artifact_to_torch
export_pyg = save_artifact_to_pyg
pack = pack_artifacts

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

# register built-in encoders
from eris import encoders as _encoders  # noqa: E402, F401
