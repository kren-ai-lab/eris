"""Structure-level feature tables for tabular machine learning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from eris.core.artifacts import StructureArtifact
from eris.exceptions import ConfigError, ErisError
from eris.pipelines.pack import read_packed_index, resolve_packed_locator

_DEFAULT_SOURCES = ("global_embedding", "pooled_attr")
_SOURCES = (*_DEFAULT_SOURCES, "graph_attr")
# Parameters that identify the individual sample rather than the configuration.
_PER_SAMPLE_PARAMETERS = frozenset({"source_artifact_fingerprint"})


class FeatureTableError(ErisError, ValueError):
    """Raised when artifacts cannot form one consistent feature table."""


ArtifactSource = StructureArtifact | str | Path


def _artifacts(artifacts: str | Path | Mapping[str, ArtifactSource]) -> list[tuple[str, ArtifactSource]]:
    if isinstance(artifacts, Mapping):
        return list(artifacts.items())
    index = Path(artifacts)
    return [(row.id, resolve_packed_locator(index, row.artifact_dir)) for row in read_packed_index(index)]


def _source(meta: dict[str, Any], requested: str | None, sample_id: str) -> str:
    names = {output["name"] for output in meta["outputs"]}
    if requested is not None:
        if requested not in names:
            raise FeatureTableError(f"{sample_id}: artifact has no {requested!r} output")
        return requested
    found = [name for name in _DEFAULT_SOURCES if name in names]
    if not found:
        raise FeatureTableError(
            f"{sample_id}: {meta['representation_name']!r} has no structure-level vector; "
            "pool residue features with eris.pool(...) or pass source='graph_attr' for graphs"
        )
    return found[0]


def _column_names(meta: dict[str, Any], source: str, width: int) -> list[str]:
    names = meta.get("feature_keys", {}).get(f"{source}_keys")
    return list(names) if names else [f"p_{i}" for i in range(width)]


def _identity(meta: dict[str, Any], source: str, columns: list[str], dtype: str) -> dict[str, Any]:
    return {
        "representation_name": meta["representation_name"],
        "coordinate_source": meta["coordinate_source"],
        "encoder": meta["encoder"],
        "parameters": {k: v for k, v in meta["parameters"].items() if k not in _PER_SAMPLE_PARAMETERS},
        "model": meta.get("model"),
        "source": source,
        "columns": columns,
        "dtype": dtype,
    }


def feature_table(
    artifacts: str | Path | Mapping[str, ArtifactSource],
    *,
    source: str | None = None,
) -> pl.DataFrame:
    """Collect one structure-level vector per artifact into an ``id`` + features DataFrame.

    ``artifacts`` is a packed index written by ``eris.pack`` or a mapping from sample id to an
    artifact or artifact directory. ``source`` defaults to ``global_embedding`` (learned) or
    ``pooled_attr`` (``eris.pool``); pass ``"graph_attr"`` for explicit graph statistics. Every
    row must share the representation, encoder, canonical parameters, model and feature names.
    Values are copied as stored, including NaN.
    """
    if source is not None and source not in _SOURCES:
        raise ConfigError(f"source must be one of: {', '.join(_SOURCES)}")
    entries = _artifacts(artifacts)
    if not entries:
        raise FeatureTableError("No artifacts to tabulate")

    ids: list[str] = []
    rows = []
    reference: dict[str, Any] | None = None
    reference_id = ""
    for sample_id, item in entries:
        artifact = item if isinstance(item, StructureArtifact) else StructureArtifact.load(item)
        row_source = _source(artifact.meta, source, sample_id)
        values = artifact.data[row_source]
        if values.ndim != 1:
            raise FeatureTableError(f"{sample_id}: {row_source!r} is not a structure-level vector")
        identity = _identity(
            artifact.meta,
            row_source,
            _column_names(artifact.meta, row_source, len(values)),
            str(values.dtype),
        )
        if reference is None:
            reference, reference_id = identity, sample_id
        elif identity != reference:
            differing = ", ".join(key for key in identity if identity[key] != reference[key])
            raise FeatureTableError(f"{sample_id}: differs from {reference_id} in {differing}")
        ids.append(sample_id)
        rows.append(values)

    features = pl.from_numpy(np.vstack(rows), schema=cast("dict[str, Any]", reference)["columns"])
    return features.insert_column(0, pl.Series("id", ids, dtype=pl.String))


__all__ = ["FeatureTableError", "feature_table"]
