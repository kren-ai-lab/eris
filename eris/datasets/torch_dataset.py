from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eris.core.artifacts import StructureArtifact
from eris.export.torch import to_torch
from eris.pipelines.pack import (
    PackedIndexRow,
    PackIndexError,
    parse_shape_summary,
    read_packed_index,
    resolve_packed_locator,
    shape_summary_from_metadata,
)


class DatasetIndexError(ValueError):
    """Raised when a packed index row cannot be consumed safely."""


def _resolve_row_path(index_path: Path, locator: str) -> Path:
    return resolve_packed_locator(index_path, locator)


def _validate_meta_locator(index_path: Path, row: PackedIndexRow) -> None:
    artifact_dir = _resolve_row_path(index_path, row.artifact_dir).resolve(strict=False)
    meta_path = _resolve_row_path(index_path, row.meta_path).resolve(strict=False)
    expected_meta_path = (artifact_dir / "meta.json").resolve(strict=False)
    if meta_path != expected_meta_path:
        raise DatasetIndexError(
            f"Packed index row {row.id!r}/{row.representation_name!r} has inconsistent "
            f"meta_path {row.meta_path!r}; expected artifact_dir/meta.json"
        )


def _counts_match(row: PackedIndexRow, artifact: StructureArtifact, field: str) -> bool:
    expected = getattr(row, field)
    if expected is None:
        return True
    counts = artifact.meta.get("counts")
    if not isinstance(counts, dict) or field not in counts:
        return False
    return counts[field] == expected


def _validate_artifact_consistency(row: PackedIndexRow, artifact: StructureArtifact) -> None:
    meta = artifact.meta
    checks = (
        ("representation_name", row.representation_name),
        ("representation_family", row.representation_family),
        ("representation_type", row.representation_type),
        ("granularity", row.granularity),
    )
    for field, expected in checks:
        actual = meta.get(field)
        if actual != expected:
            raise DatasetIndexError(
                f"Packed index row {row.id!r}/{row.representation_name!r} is stale: "
                f"{field}={expected!r} does not match artifact {actual!r}"
            )

    for field in ("n_nodes", "n_edges"):
        if not _counts_match(row, artifact, field):
            raise DatasetIndexError(
                f"Packed index row {row.id!r}/{row.representation_name!r} is stale: "
                f"{field} does not match artifact counts"
            )

    if parse_shape_summary(row.shape_summary) != json.loads(shape_summary_from_metadata(meta)):
        raise DatasetIndexError(
            f"Packed index row {row.id!r}/{row.representation_name!r} is stale: "
            "shape_summary does not match artifact shapes"
        )


def _learned_sample(artifact: StructureArtifact) -> dict[str, Any]:
    return dict(artifact.data)


@dataclass
class ErisFolderDataset:
    """
    Dataset consumer for canonical ``eris pack`` indexes.

    The packed index is the collection snapshot. Artifact locators are POSIX-style paths
    relative to the index file parent. Individual artifacts are strict-loaded lazily in
    ``__getitem__``.
    """

    index_path: str | Path
    device: str | None = None
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    matrix_as_channel: bool = True

    def __post_init__(self) -> None:
        self.index_path = Path(self.index_path)
        try:
            self.rows = read_packed_index(self.index_path)
        except PackIndexError as error:
            raise DatasetIndexError(str(error)) from error
        for row in self.rows:
            _validate_meta_locator(self.index_path, row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        artifact_dir = _resolve_row_path(self.index_path, row.artifact_dir)
        artifact = StructureArtifact.load(artifact_dir, strict=True)
        _validate_artifact_consistency(row, artifact)

        if row.representation_family == "learned":
            sample = _learned_sample(artifact)
        else:
            try:
                sample = to_torch(
                    artifact,
                    device=self.device,
                    matrix_as_channel=self.matrix_as_channel,
                )
            except ImportError as error:
                raise RuntimeError(
                    "ErisFolderDataset explicit tensor conversion requires optional "
                    "dependency 'torch'. Install the Torch/export dependencies to load "
                    "explicit artifacts as tensors."
                ) from error

        sample["id"] = row.id
        sample["artifact_dir"] = str(artifact_dir)
        sample["meta_path"] = str(_resolve_row_path(self.index_path, row.meta_path))
        sample["representation_name"] = row.representation_name
        sample["representation_family"] = row.representation_family
        sample["representation_type"] = row.representation_type
        sample["granularity"] = row.granularity
        for key in ("parameters", "feature_keys", "outputs"):
            if key in artifact.meta:
                sample[key] = deepcopy(artifact.meta[key])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


__all__ = ["DatasetIndexError", "ErisFolderDataset"]
