from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import eris
from eris.core.artifact_schema import compute_config_fingerprint
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.cache import parser_hint_from_source, parser_hint_matches_format
from eris.core.config import CanonicalRepresentationConfig
from eris.core.hashing import sha256_file
from eris.core.io import resolve_structure_input
from eris.core.registry import get_encoder, get_encoder_info, resolve_representation_config
from eris.core.types import StructureInput
from eris.exceptions import ConfigError, ExecutionError
from eris.pipelines.manifest import (
    BatchResultRow,
    StructureManifestRow,
    read_manifest,
    write_atomically,
    write_manifest,
)

RepresentationFamily = Literal["explicit", "learned"]


@dataclass(frozen=True)
class _ResumeIdentity:
    representation_name: str
    representation_family: RepresentationFamily
    representation_type: str
    granularity: str
    coordinate_source: str
    parameters: dict[str, Any]
    config_fingerprint: str
    chain_ids: tuple[str, ...]
    model: dict[str, Any] | None = None
    encoder_implementation: str | None = None
    encoder_version: str | None = None
    execution: dict[str, str] | None = None


def _coerce_bool(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{name} must be a bool")
    return value


def _coerce_rows(
    rows: Iterable[StructureManifestRow] | str | Path | None,
    *,
    manifest_path: str | Path | None,
) -> list[StructureManifestRow]:
    if rows is not None and manifest_path is not None:
        raise ConfigError("rows and manifest_path cannot be combined")
    source = manifest_path if manifest_path is not None else rows
    if source is None:
        raise ConfigError("rows or manifest_path is required")
    if isinstance(source, (str, Path)):
        return read_manifest(Path(source), StructureManifestRow)

    materialized = list(source)
    if not materialized:
        raise ConfigError("batch rows must be non-empty")
    for index, row in enumerate(materialized, start=1):
        if not isinstance(row, StructureManifestRow):
            raise TypeError(f"row {index} must be a StructureManifestRow")
    return materialized


def _resolve_representation(representation: str) -> str:
    if not isinstance(representation, str) or not representation.strip():
        raise ConfigError("representation must be a non-empty string")
    return representation.strip()


def _resolve_artifact_root(
    artifact_root: str | Path | None,
) -> Path:
    if artifact_root is None:
        raise ConfigError("artifact_root is required")
    if not isinstance(artifact_root, (str, Path)):
        raise TypeError("artifact_root must be a string or pathlib.Path")
    return Path(artifact_root)


def _normalize_explicit_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping or None")
    return dict(params)


def _explicit_resume_identity(
    representation: str,
    config: CanonicalRepresentationConfig,
    chain_ids: tuple[str, ...],
) -> _ResumeIdentity:
    info = get_encoder_info(representation)
    encoder = get_encoder(representation, config=config, use_cache=False)
    parameters = config.to_fingerprint_dict()
    return _ResumeIdentity(
        representation_name=representation,
        representation_family="explicit",
        representation_type=info.representation_type,
        granularity=info.granularity,
        coordinate_source=config.coordinate_source,
        parameters=parameters,
        config_fingerprint=compute_config_fingerprint(representation, parameters, chain_ids=chain_ids),
        chain_ids=chain_ids,
        encoder_implementation=encoder.encoder_implementation,
        encoder_version=encoder.version,
    )


def _resolve_row_explicit_configs(
    rows: list[StructureManifestRow],
    *,
    representation: str,
    params: Mapping[str, Any] | None,
    params_by_id: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, CanonicalRepresentationConfig]:
    global_params = _normalize_explicit_params(params)
    if params_by_id is None:
        row_overrides: dict[str, Mapping[str, Any]] = {}
    else:
        if not isinstance(params_by_id, Mapping):
            raise TypeError("params_by_id must be a mapping or None")
        if any(not isinstance(row_id, str) for row_id in params_by_id):
            raise TypeError("params_by_id keys must be structure row IDs")
        row_ids = {row.id for row in rows}
        unknown_ids = sorted(set(params_by_id) - row_ids)
        if unknown_ids:
            raise ConfigError(f"params_by_id contains unknown row IDs: {unknown_ids}")
        invalid_ids = [
            row_id for row_id, override in params_by_id.items() if not isinstance(override, Mapping)
        ]
        if invalid_ids:
            raise TypeError(f"params_by_id values must be mappings; invalid row IDs: {sorted(invalid_ids)}")
        row_overrides = dict(params_by_id)

    config_by_id: dict[str, CanonicalRepresentationConfig] = {}
    for row in rows:
        effective = dict(global_params)
        effective.update(row_overrides.get(row.id, {}))
        config_by_id[row.id] = resolve_representation_config(
            representation,
            params=effective,
        )
    return config_by_id


def _learned_resume_identity(
    representation: str,
    *,
    output_mode: str,
    device: str,
    precision: str,
    chain_ids: tuple[str, ...],
) -> _ResumeIdentity:
    from eris.core.learned.execution import (
        StructuralExecutionConfig,
        execution_request_cache_identity,
    )
    from eris.core.learned.registry import learned_registry

    info = eris.get_embedder_info(representation)
    registry_info = learned_registry.get_info(representation)
    config = registry_info.config_type(
        representation_name=registry_info.representation_name, output_mode=output_mode
    )
    execution = StructuralExecutionConfig(device=device, precision=precision)
    parameters = config.to_dict()
    return _ResumeIdentity(
        representation_name=representation,
        representation_family="learned",
        representation_type=info["type"],
        granularity=info["granularity_by_output_mode"][output_mode],
        coordinate_source=info["coordinate_source"],
        parameters=parameters,
        config_fingerprint=compute_config_fingerprint(representation, parameters, chain_ids=chain_ids),
        chain_ids=chain_ids,
        model=registry_info.model_descriptor.to_cache_identity(),
        encoder_implementation=registry_info.implementation,
        encoder_version=registry_info.encoder_version,
        execution=execution_request_cache_identity(execution),
    )


def _target_dir(artifact_root: Path, row: StructureManifestRow, representation: str) -> Path:
    return artifact_root / row.id / representation


def _resolved_row_input(row: StructureManifestRow) -> StructureInput:
    return resolve_structure_input(StructureInput(source=row.path, chain_id=row.chain, model_index=row.model))


def _staging_dir(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))


def _cleanup_staging(staging: Path, target: Path) -> None:
    if staging.exists():
        resolved_parent = staging.parent.resolve()
        if resolved_parent != target.parent.resolve() or not staging.name.startswith(f".{target.name}.tmp-"):
            raise RuntimeError(f"Refusing to remove unexpected staging path: {staging}")
        shutil.rmtree(staging)


def _publish_staging(staging: Path, target: Path) -> None:
    StructureArtifact.load(staging, strict=True)
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"Existing artifact path is not a directory: {target}")
    if not target.exists():
        os.replace(staging, target)
        return

    for filename in ("data.npz", "meta.json"):
        os.replace(staging / filename, target / filename)
    _cleanup_staging(staging, target)


def _save_artifact_safely(artifact: StructureArtifact, target: Path) -> None:
    staging = _staging_dir(target)
    published = False
    try:
        artifact.save(staging)
        _publish_staging(staging, target)
        published = True
        StructureArtifact.load(target, strict=True)
    finally:
        if not published:
            _cleanup_staging(staging, target)


def _write_results_safely(path: Path, rows: list[BatchResultRow]) -> None:
    write_atomically(path, lambda staging: write_manifest(staging, rows))


def _input_matches(
    artifact: StructureArtifact,
    row: StructureManifestRow,
    *,
    source_sha256: str,
    chain_ids: tuple[str, ...],
) -> bool:
    input_meta = artifact.meta.get("input")
    if not isinstance(input_meta, dict):
        return False
    input_format = input_meta.get("format")
    return (
        input_meta.get("source_sha256") == source_sha256
        and input_meta.get("model_index") == row.model
        and input_meta.get("chain_ids") == list(chain_ids)
        and isinstance(input_format, str)
        and parser_hint_matches_format(parser_hint_from_source(row.path), input_format)
    )


def _artifact_matches_request(
    artifact: StructureArtifact,
    row: StructureManifestRow,
    identity: _ResumeIdentity,
    *,
    source_sha256: str,
) -> bool:
    try:
        artifact.validate(strict=True)
    except (ArtifactValidationError, TypeError, ValueError):
        return False

    meta = artifact.meta
    provenance = meta.get("provenance")
    encoder = meta.get("encoder")
    if not isinstance(provenance, dict) or not isinstance(encoder, dict):
        return False

    matches = (
        meta.get("representation_name") == identity.representation_name
        and meta.get("representation_family") == identity.representation_family
        and meta.get("representation_type") == identity.representation_type
        and meta.get("granularity") == identity.granularity
        and meta.get("coordinate_source") == identity.coordinate_source
        and meta.get("parameters") == identity.parameters
        and provenance.get("config_fingerprint") == identity.config_fingerprint
        and _input_matches(
            artifact,
            row,
            source_sha256=source_sha256,
            chain_ids=identity.chain_ids,
        )
    )
    if not matches:
        return False
    if identity.model is not None:
        from eris.core.learned.runner import _cached_model_identity

        if _cached_model_identity(meta.get("model")) != identity.model:
            return False
    if identity.model is None and "model" in meta:
        return False
    if identity.encoder_implementation is not None:
        if encoder.get("implementation") != identity.encoder_implementation:
            return False
    if identity.encoder_version is not None:
        if encoder.get("version") != identity.encoder_version:
            return False
    if identity.execution is None:
        return True
    from eris.core.learned.runner import _cached_execution_matches

    return _cached_execution_matches(
        meta.get("provenance"), requested_precision=identity.execution["precision"]
    )


def _load_equivalent_artifact(
    target: Path,
    row: StructureManifestRow,
    identity: _ResumeIdentity,
) -> StructureArtifact | None:
    try:
        source_sha256 = sha256_file(Path(row.path))
        artifact = StructureArtifact.load(target, strict=True)
    except (ArtifactValidationError, OSError):
        return None
    if _artifact_matches_request(
        artifact,
        row,
        identity,
        source_sha256=source_sha256,
    ):
        return artifact
    return None


def _success_row(
    row: StructureManifestRow,
    *,
    representation: str,
    family: RepresentationFamily,
    artifact_dir: Path,
) -> BatchResultRow:
    return BatchResultRow(
        id=row.id,
        representation_name=representation,
        representation_family=family,
        artifact_dir=str(artifact_dir),
        status="success",
        error_type=None,
        error_message=None,
    )


def _failure_row(
    row: StructureManifestRow,
    *,
    representation: str,
    family: RepresentationFamily,
    error: Exception,
) -> BatchResultRow:
    return BatchResultRow(
        id=row.id,
        representation_name=representation,
        representation_family=family,
        artifact_dir=None,
        status="failed",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _run_batch(
    rows: list[StructureManifestRow],
    *,
    representation: str,
    family: RepresentationFamily,
    artifact_root: Path,
    resume: bool,
    overwrite: bool,
    fail_fast: bool,
    results_path: str | Path | None,
    identity_factory: Callable[[StructureManifestRow], _ResumeIdentity],
    execute: Callable[[StructureManifestRow], StructureArtifact],
) -> list[BatchResultRow]:
    _coerce_bool(resume, "resume")
    _coerce_bool(overwrite, "overwrite")
    _coerce_bool(fail_fast, "fail_fast")

    artifact_root.mkdir(parents=True, exist_ok=True)
    results_file = Path(results_path) if results_path is not None else None
    results: list[BatchResultRow] = []

    for row in rows:
        target = _target_dir(artifact_root, row, representation)
        try:
            if target.exists() and not resume and not overwrite:
                raise FileExistsError(
                    f"Output artifact already exists: {target}; use resume=True "
                    "to reuse equivalent artifacts or overwrite=True to replace it"
                )

            if resume and target.exists():
                identity = identity_factory(row)
                artifact = _load_equivalent_artifact(target, row, identity)
                if artifact is not None:
                    results.append(
                        _success_row(
                            row,
                            representation=representation,
                            family=family,
                            artifact_dir=target,
                        )
                    )
                    if results_file is not None:
                        _write_results_safely(results_file, results)
                    continue

            artifact = execute(row)
            if not isinstance(artifact, StructureArtifact):
                raise ExecutionError("public execution did not return StructureArtifact")
            artifact.validate(strict=True)
            _save_artifact_safely(artifact, target)
            results.append(
                _success_row(
                    row,
                    representation=representation,
                    family=family,
                    artifact_dir=target,
                )
            )
        except Exception as error:
            results.append(
                _failure_row(
                    row,
                    representation=representation,
                    family=family,
                    error=error,
                )
            )
            if results_file is not None:
                _write_results_safely(results_file, results)
            if fail_fast:
                raise
        else:
            if results_file is not None:
                _write_results_safely(results_file, results)

    return results


def batch_encode(
    rows: Iterable[StructureManifestRow] | str | Path | None = None,
    *,
    representation: str,
    artifact_root: str | Path,
    params: Mapping[str, Any] | None = None,
    params_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    use_cache: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    fail_fast: bool = False,
    results_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> list[BatchResultRow]:
    """Run one explicit representation with optional shallow per-row overrides."""
    representation_name = _resolve_representation(representation)
    artifact_root_path = _resolve_artifact_root(artifact_root)
    structure_rows = _coerce_rows(rows, manifest_path=manifest_path)
    explicit_configs_by_id = _resolve_row_explicit_configs(
        structure_rows,
        representation=representation_name,
        params=params,
        params_by_id=params_by_id,
    )
    _coerce_bool(use_cache, "use_cache")

    return _run_batch(
        structure_rows,
        representation=representation_name,
        family="explicit",
        artifact_root=artifact_root_path,
        resume=resume,
        overwrite=overwrite,
        fail_fast=fail_fast,
        results_path=results_path,
        identity_factory=lambda row: _explicit_resume_identity(
            representation_name,
            explicit_configs_by_id[row.id],
            _resolved_row_input(row).chain_id,
        ),
        execute=lambda row: eris.encode(
            row.path,
            representation=representation_name,
            chain_id=row.chain,
            model_index=row.model,
            config=explicit_configs_by_id[row.id],
            use_cache=use_cache,
        ),
    )


def batch_embed(
    rows: Iterable[StructureManifestRow] | str | Path | None = None,
    *,
    representation: str,
    artifact_root: str | Path,
    output_mode: str = "both",
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    precision: str = "fp32",
    use_cache: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    fail_fast: bool = False,
    results_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> list[BatchResultRow]:
    """Run one learned structural embedder over a structure manifest."""
    if not isinstance(representation, str) or not representation.strip():
        raise ConfigError("representation must be a non-empty string")
    _coerce_bool(use_cache, "use_cache")

    representation_name = representation.strip()
    artifact_root_path = _resolve_artifact_root(artifact_root)
    structure_rows = _coerce_rows(rows, manifest_path=manifest_path)

    return _run_batch(
        structure_rows,
        representation=representation_name,
        family="learned",
        artifact_root=artifact_root_path,
        resume=resume,
        overwrite=overwrite,
        fail_fast=fail_fast,
        results_path=results_path,
        identity_factory=lambda row: _learned_resume_identity(
            representation_name,
            output_mode=output_mode,
            device=device,
            precision=precision,
            chain_ids=_resolved_row_input(row).chain_id,
        ),
        execute=lambda row: eris.embed(
            row.path,
            representation=representation_name,
            chain_id=row.chain,
            model_index=row.model,
            output_mode=output_mode,
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            use_cache=use_cache,
        ),
    )


__all__ = ["batch_embed", "batch_encode"]
