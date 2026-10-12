"""Export of artifacts to PyTorch Geometric data."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eris.core.artifacts import StructureArtifact
from eris.export.torch import to_torch


class PyGExportError(ValueError):
    """Raised when an artifact cannot be exported as a PyG graph."""


_METADATA_ATTRS = (
    "schema_version",
    "representation_name",
    "representation_family",
    "representation_type",
    "granularity",
    "coordinate_source",
    "feature_keys",
    "parameters",
    "outputs",
)
_EDGE_INDEX_INTEGER_DTYPES = {
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.uint8",
}


def _pyg_dependencies() -> tuple[Any, Any]:
    try:
        import torch  # noqa: PLC0415
        from torch_geometric.data import Data  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyG export requires optional dependencies 'torch' and 'torch_geometric'. "
            "Install the PyG/export dependencies before calling eris.export.pyg.to_pyg()."
        ) from error

    return torch, Data


def _metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(meta[key]) for key in _METADATA_ATTRS if key in meta}


def _identity(artifact: StructureArtifact) -> tuple[str, str]:
    artifact.validate(strict=True)
    family = artifact.meta.get("representation_family")
    representation_type = artifact.meta.get("representation_type")
    if not isinstance(family, str) or not isinstance(representation_type, str):
        raise PyGExportError("Artifact metadata must define representation_family and representation_type.")
    return family, representation_type


def _ensure_graph_artifact(artifact: StructureArtifact) -> None:
    family, representation_type = _identity(artifact)
    if family == "learned":
        raise PyGExportError(
            "Learned artifacts do not expose public graph topology for PyG graph export; "
            "use Torch export for learned embeddings."
        )
    if family != "explicit":
        raise PyGExportError(f"Unsupported representation_family={family!r} for PyG export.")
    if representation_type != "graph":
        raise PyGExportError(
            f"Unsupported representation_type={representation_type!r} for PyG graph export; "
            "only explicit graph artifacts with public edge_index topology are supported."
        )
    if "edge_index" not in artifact.data:
        raise PyGExportError("Graph artifact is missing required public topology 'edge_index'.")


def _node_id_output_name(meta: Mapping[str, Any]) -> str | None:
    outputs = meta.get("outputs")
    if not isinstance(outputs, list):
        return None
    for output in outputs:
        if (
            isinstance(output, Mapping)
            and output.get("role") == "node_ids"
            and isinstance(output.get("name"), str)
        ):
            return output["name"]
    return None


def _as_lossless_list(value: Any) -> list[Any]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [array.item()]
    return array.reshape(-1).tolist()


def _attach_common_attrs(data: Any, metadata: Mapping[str, Any]) -> None:
    for key, value in metadata.items():
        setattr(data, key, value)


def _attach_auxiliary_attrs(data: Any, torch_mapping: Mapping[str, Any]) -> None:
    extra = torch_mapping.get("extra")
    if isinstance(extra, Mapping):
        data.extra = deepcopy(dict(extra))


def _torch_dict_to_pyg(out: Mapping[str, Any], *, device: str | None = None) -> Any:
    """Convert a canonical graph Torch mapping into ``torch_geometric.data.Data``.

    This compatibility helper is graph-only. It does not infer topology for cloud,
    matrix, grid, or learned embedding mappings.
    """
    family = out.get("representation_family")
    representation_type = out.get("representation_type")
    if family is not None and family != "explicit":
        raise PyGExportError("Only explicit graph Torch mappings are supported for PyG graph export.")
    if representation_type is not None and representation_type != "graph":
        raise PyGExportError("Only graph Torch mappings are supported for PyG graph export.")
    if "edge_index" not in out:
        raise PyGExportError("PyG graph export requires public 'edge_index' topology.")

    torch, Data = _pyg_dependencies()
    pyg = Data()
    edge_index = out["edge_index"]
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index).clone()
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise PyGExportError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
    if str(edge_index.dtype) not in _EDGE_INDEX_INTEGER_DTYPES:
        raise PyGExportError("edge_index must use an integer dtype.")
    pyg.edge_index = edge_index

    for source_key, target_key in (
        ("x", "x"),
        ("pos", "pos"),
        ("edge_attr", "edge_attr"),
        ("edge_dist", "edge_dist"),
        ("graph_attr", "graph_attr"),
        ("centroid", "centroid"),
    ):
        value: Any = out.get(source_key)
        if isinstance(value, torch.Tensor):
            setattr(pyg, target_key, value.clone())

    _attach_common_attrs(
        pyg,
        {key: deepcopy(out[key]) for key in _METADATA_ATTRS if key in out},
    )
    _attach_auxiliary_attrs(pyg, out)

    node_ids = out.get("node_ids")
    if node_ids is not None:
        pyg.node_ids = deepcopy(node_ids)

    if device is not None:
        pyg = pyg.to(device)
    return pyg


def to_pyg(
    artifact: StructureArtifact,
    *,
    device: str | None = None,
) -> Any:
    """Export one explicit graph artifact schema artifact to ``torch_geometric.data.Data``.

    Non-graph artifacts are rejected because this exporter preserves public topology; it
    never invents edges from clouds, matrices, grids, or learned model internals.
    """
    if not isinstance(artifact, StructureArtifact):
        raise TypeError("to_pyg() expects a StructureArtifact instance.")
    _ensure_graph_artifact(artifact)

    torch_mapping = to_torch(artifact)
    node_id_name = _node_id_output_name(artifact.meta)
    if node_id_name is not None and node_id_name in artifact.data:
        torch_mapping["node_ids"] = _as_lossless_list(artifact.data[node_id_name])
    pyg = _torch_dict_to_pyg(torch_mapping, device=device)
    _attach_common_attrs(pyg, _metadata(artifact.meta))
    return pyg


def save_artifact_to_pyg(
    artifact_dir: str | Path,
    *,
    out_path: Path,
    device: str | None = None,
) -> None:
    """Export one artifact directory to a PyG file."""
    torch, _ = _pyg_dependencies()
    artifact = StructureArtifact.load(Path(artifact_dir), strict=True)
    pyg = to_pyg(artifact, device=device)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pyg, out_path)


__all__ = [
    "PyGExportError",
    "save_artifact_to_pyg",
    "to_pyg",
]
