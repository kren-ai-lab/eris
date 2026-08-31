from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eris.core.artifacts import StructureArtifact


class TorchExportError(ValueError):
    """Raised when an Artifact Schema v2 artifact cannot be exported to Torch."""


_COMMON_METADATA = (
    "representation_name",
    "representation_family",
    "representation_type",
    "granularity",
    "feature_keys",
    "parameters",
    "outputs",
)
_MATRIX_KEYS = ("dist", "dist_bins", "contact", "contact_bins")
_AUXILIARY_KEYS = (
    "residue_ids",
    "resname",
    "aa_index",
    "plddt",
    "bfactor_ca",
    "atom_name",
    "elem_index",
    "atom_role",
    "bfactor",
    "occ",
    "seq_pos_residue",
    "aa_index_residue",
)


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Torch export requires optional dependency 'torch'. Install the Torch/export "
            "dependencies before calling eris.export.torch.to_torch()."
        ) from error

    return torch


def _to_tensor(value: Any):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()

    array = np.asarray(value)
    if not array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, copy=True)
    else:
        array = array.copy()
    return torch.as_tensor(array)


def _require_array(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise TorchExportError(f"Artifact data is missing required output {key!r}")
    return data[key]


def _metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(meta[key]) for key in _COMMON_METADATA if key in meta}


def _output_names(meta: Mapping[str, Any]) -> list[str]:
    outputs = meta.get("outputs")
    if not isinstance(outputs, list):
        raise TorchExportError("Artifact metadata must contain Schema v2 outputs.")
    names: list[str] = []
    for index, output in enumerate(outputs):
        if not isinstance(output, Mapping) or not isinstance(output.get("name"), str):
            raise TorchExportError(f"Artifact metadata outputs[{index}] has no valid name.")
        names.append(output["name"])
    return names


def _matrix_key(data: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    for name in _output_names(meta):
        if name in _MATRIX_KEYS:
            return name
    for name in _MATRIX_KEYS:
        if name in data:
            return name
    raise TorchExportError(
        "Matrix artifact is missing one of 'dist', 'dist_bins', 'contact', or 'contact_bins'."
    )


def _attach_extra(out: dict[str, Any], data: Mapping[str, Any]) -> None:
    extra: dict[str, Any] = {}
    for key in _AUXILIARY_KEYS:
        if key not in data:
            continue
        array = np.asarray(data[key])
        if array.dtype.kind in {"U", "S", "O"}:
            extra[key] = array.reshape(-1).tolist()
        else:
            extra[key] = _to_tensor(array)
    if extra:
        out["extra"] = extra


def _graph_to_torch(artifact: StructureArtifact) -> dict[str, Any]:
    data = artifact.data
    out = _metadata(artifact.meta)
    node_attr = _to_tensor(_require_array(data, "node_attr"))
    edge_index = _to_tensor(_require_array(data, "edge_index"))
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise TorchExportError(
            f"Graph edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        )
    if edge_index.dtype.is_floating_point or edge_index.dtype == _torch().bool:
        raise TorchExportError("Graph edge_index must use an integer dtype.")

    out["x"] = node_attr
    out["edge_index"] = edge_index

    for key in ("edge_attr", "edge_dist", "pos", "graph_attr", "centroid"):
        if key in data:
            value = _to_tensor(data[key])
            out[key] = value.reshape(-1) if key == "graph_attr" else value

    _attach_extra(out, data)
    return out


def _cloud_to_torch(artifact: StructureArtifact) -> dict[str, Any]:
    data = artifact.data
    out = _metadata(artifact.meta)
    out["pos"] = _to_tensor(_require_array(data, "pos"))
    if "node_attr" in data:
        out["x"] = _to_tensor(data["node_attr"])
    if "centroid" in data:
        out["centroid"] = _to_tensor(data["centroid"])
    _attach_extra(out, data)
    return out


def _matrix_to_torch(
    artifact: StructureArtifact,
    *,
    matrix_as_channel: bool,
) -> dict[str, Any]:
    data = artifact.data
    out = _metadata(artifact.meta)
    key = _matrix_key(data, artifact.meta)
    matrix = _to_tensor(_require_array(data, key))
    if matrix.ndim != 2:
        raise TorchExportError(f"Matrix key {key!r} must be 2D, got {tuple(matrix.shape)}")
    out["x"] = matrix.unsqueeze(0) if matrix_as_channel else matrix
    out["matrix_key"] = key
    out["n"] = int(matrix.shape[0])

    for optional in ("bin_edges", "pos", "centroid"):
        if optional in data:
            out[optional] = _to_tensor(data[optional])

    _attach_extra(out, data)
    return out


def _grid_to_torch(artifact: StructureArtifact) -> dict[str, Any]:
    data = artifact.data
    out = _metadata(artifact.meta)
    grid = _to_tensor(_require_array(data, "grid"))
    out["x"] = grid

    for optional in ("grid_size", "voxel_size", "origin", "centroid"):
        if optional in data:
            out[optional] = _to_tensor(data[optional])

    _attach_extra(out, data)
    return out


def _as_flat_python_list(value: Any, *, key: str) -> list[Any]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [array.item()]
    return array.reshape(-1).tolist()


def _learned_output_layout(data: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    names = set(_output_names(meta))
    has_node = "node_embedding" in data or "node_embedding" in names
    has_node_ids = "node_ids" in data or "node_ids" in names
    has_global = "global_embedding" in data or "global_embedding" in names
    has_valid_mask = "valid_mask" in data or "valid_mask" in names

    if has_node != has_node_ids:
        raise TorchExportError("Learned node export requires both 'node_embedding' and 'node_ids'.")
    if has_valid_mask and not has_node:
        raise TorchExportError("Learned valid_mask requires node_embedding and node_ids.")
    if has_node and has_global:
        return "both"
    if has_node:
        return "node"
    if has_global:
        return "global"
    raise TorchExportError("Learned artifact has no recognized embedding outputs.")


def _learned_to_torch(artifact: StructureArtifact) -> dict[str, Any]:
    data = artifact.data
    out = _metadata(artifact.meta)
    layout = _learned_output_layout(data, artifact.meta)

    if layout in {"node", "both"}:
        node_embedding = _to_tensor(_require_array(data, "node_embedding"))
        if node_embedding.ndim != 2:
            raise TorchExportError(
                f"Learned node_embedding must have shape (N, D), got {tuple(node_embedding.shape)}"
            )
        node_ids = _as_flat_python_list(_require_array(data, "node_ids"), key="node_ids")
        if len(node_ids) != int(node_embedding.shape[0]):
            raise TorchExportError("Learned node_ids length must match node_embedding row count.")
        out["node_embedding"] = node_embedding
        out["node_ids"] = node_ids

        if "valid_mask" in data:
            valid_mask = _to_tensor(data["valid_mask"])
            if valid_mask.ndim != 1:
                raise TorchExportError(
                    f"Learned valid_mask must be 1D, got {tuple(valid_mask.shape)}"
                )
            if valid_mask.dtype != _torch().bool:
                raise TorchExportError("Learned valid_mask must use bool dtype.")
            if int(valid_mask.shape[0]) != int(node_embedding.shape[0]):
                raise TorchExportError(
                    "Learned valid_mask length must match node_embedding row count."
                )
            out["valid_mask"] = valid_mask

    if layout in {"global", "both"}:
        global_embedding = _to_tensor(_require_array(data, "global_embedding"))
        if global_embedding.ndim != 1:
            raise TorchExportError(
                "Learned global_embedding must have shape (D,), "
                f"got {tuple(global_embedding.shape)}"
            )
        out["global_embedding"] = global_embedding

    return out


def _representation_identity(artifact: StructureArtifact) -> tuple[str, str]:
    meta = artifact.meta
    family = meta.get("representation_family")
    representation_type = meta.get("representation_type")
    if not isinstance(family, str) or not isinstance(representation_type, str):
        raise TorchExportError(
            "Artifact metadata must define representation_family and representation_type."
        )
    return family, representation_type


def _apply_device(out: dict[str, Any], device: str | None) -> dict[str, Any]:
    if device is None:
        return out
    torch = _torch()
    for key, value in list(out.items()):
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif key == "extra" and isinstance(value, Mapping):
            out[key] = {
                extra_key: extra_value.to(device)
                if isinstance(extra_value, torch.Tensor)
                else extra_value
                for extra_key, extra_value in value.items()
            }
    return out


def to_torch(
    artifact: StructureArtifact,
    *,
    prefer: str = "grid_or_graph",
    device: str | None = None,
    matrix_as_channel: bool = True,
) -> dict[str, Any]:
    """
    Convert one strict-valid Artifact Schema v2 ``StructureArtifact`` to Torch.

    The returned mapping contains Torch tensors plus lightweight metadata. Torch export is
    downstream-only: it does not mutate artifacts, change canonical storage, batch samples,
    invent topology, or invoke learned backends.
    """
    if not isinstance(artifact, StructureArtifact):
        raise TypeError("to_torch() expects a StructureArtifact instance.")

    artifact.validate(strict=True)
    family, representation_type = _representation_identity(artifact)
    prefer = (prefer or "grid_or_graph").strip().lower()
    if prefer not in {"grid_or_graph", "grid", "graph", "matrix", "cloud"}:
        raise ValueError("prefer must be one of: grid_or_graph, grid, graph, matrix, cloud")

    if family == "learned":
        if representation_type != "embedding":
            raise TorchExportError(
                f"Unsupported learned representation_type={representation_type!r}."
            )
        return _apply_device(_learned_to_torch(artifact), device)

    if family != "explicit":
        raise TorchExportError(f"Unsupported representation_family={family!r}.")

    if representation_type == "graph":
        if prefer not in {"grid_or_graph", "graph"}:
            raise TorchExportError(f"Cannot export graph artifact with prefer={prefer!r}.")
        out = _graph_to_torch(artifact)
    elif representation_type == "cloud":
        if prefer not in {"grid_or_graph", "cloud"}:
            raise TorchExportError(f"Cannot export cloud artifact with prefer={prefer!r}.")
        out = _cloud_to_torch(artifact)
    elif representation_type == "matrix":
        if prefer not in {"grid_or_graph", "matrix"}:
            raise TorchExportError(f"Cannot export matrix artifact with prefer={prefer!r}.")
        out = _matrix_to_torch(artifact, matrix_as_channel=matrix_as_channel)
    elif representation_type == "grid":
        if prefer not in {"grid_or_graph", "grid"}:
            raise TorchExportError(f"Cannot export grid artifact with prefer={prefer!r}.")
        out = _grid_to_torch(artifact)
    else:
        raise TorchExportError(f"Unsupported explicit representation_type={representation_type!r}.")

    return _apply_device(out, device)


def load_torch(
    artifact_dir: str | Path,
    *,
    prefer: str = "grid_or_graph",
    device: str | None = None,
    matrix_as_channel: bool = True,
) -> dict[str, Any]:
    artifact = StructureArtifact.load(Path(artifact_dir), strict=True)
    return to_torch(
        artifact,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )


def save_artifact_to_torch(
    artifact_dir: str | Path,
    *,
    out_path: Path,
    prefer: str = "grid_or_graph",
    device: str | None = None,
    matrix_as_channel: bool = True,
) -> None:
    torch = _torch()
    artifact = StructureArtifact.load(Path(artifact_dir), strict=True)
    data = to_torch(
        artifact,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)


__all__ = [
    "TorchExportError",
    "load_torch",
    "save_artifact_to_torch",
    "to_torch",
]
