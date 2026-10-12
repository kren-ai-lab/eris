"""Shared encoder contract: context, caching and artifact assembly."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris import exceptions as exc
from eris.core import io as structure_io
from eris.core.artifact_schema import (
    build_artifact_metadata,
    build_input_metadata,
    build_provenance,
    compute_config_fingerprint,
)
from eris.core.artifacts import StructureArtifact
from eris.core.cache import (
    DiskCache,
    compute_cache_key,
    parser_hint_from_source,
    parser_hint_matches_format,
)
from eris.core.hashing import sha256_file
from eris.core.types import _chain_ids

if TYPE_CHECKING:
    from eris.core.config import CanonicalRepresentationConfig, _CommonGeometryConfig
    from eris.core.io import LoadedStructureData
    from eris.core.representations.output_declarations import OutputDeclarations
    from eris.core.types import StructureInput

logger = logging.getLogger(__name__)


def _check_output_values(data: dict[str, np.ndarray]) -> None:
    """Reject values that come from the input rather than our code; schema validation checks the shapes."""
    if len(data["residue_ids"]) == 0:
        raise exc.ExecutionError("Encoder output contains no atoms/residues")
    for key in ("pos", "centroid", "grid"):
        if key in data and not np.isfinite(data[key]).all():
            raise ValueError(f"Encoder output {key!r} contains non-finite values")
    if "edge_attr" in data and np.isinf(data["edge_attr"]).any():
        raise ValueError("Encoder output 'edge_attr' contains infinite values")
    edge_index = data.get("edge_index")
    if (
        edge_index is not None
        and edge_index.size > 0
        and (edge_index.min() < 0 or edge_index.max() >= len(data["pos"]))
    ):
        raise ValueError("Encoder output 'edge_index' contains out-of-range node indices")


def _structure_validation(ctx: EncodingContext) -> None:
    strc = ctx.raw_structure
    assert strc is not None  # noqa: S101 - _prepare_data always populates it first
    n = int(strc["pos"].shape[0])

    if n < 20:
        logger.warning(
            "Low node count. Structure loaded.",
            extra={"n_nodes": n},
        )

    else:
        logger.info(
            "Structure loaded.",
            extra={"n_nodes": n},
        )


@dataclass
class EncodingContext:
    """Mutable per-encode state shared between the base encoder and concrete encoders."""

    inp: StructureInput
    source_sha256: str | None = None
    input_format: str | None = None
    raw_structure: LoadedStructureData | None = None
    centroid: np.ndarray | None = None
    pos: np.ndarray | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    structure: Any = None  # parsed gemmi.Structure, reused by _load_structure


class BaseEncoder(ABC):
    """Base class for all encoders.

    Defines the contract for each subclass.
    """

    # Identity
    version: ClassVar[str] = "0.1.0"
    name: ClassVar[str]
    representation_family: ClassVar[str] = "explicit"
    granularity: ClassVar[str]
    representation: ClassVar[Any]
    output_declarations: ClassVar[OutputDeclarations]
    config_type: ClassVar[type[_CommonGeometryConfig]]
    config: _CommonGeometryConfig

    def __init__(
        self,
        *,
        config: CanonicalRepresentationConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
        """Build the encoder from a canonical config or from representation keyword arguments."""
        if config is None:
            config = self.config_type(**representation_kwargs)
        elif not isinstance(config, self.config_type):
            raise TypeError(f"config must be of type {self.config_type.__name__}")
        elif representation_kwargs:
            names = ", ".join(sorted(representation_kwargs))
            raise TypeError(f"config cannot be combined with representation kwargs: {names}")
        self.config = config
        self.use_cache = use_cache
        self.cache = cache

    @property
    def encoder_implementation(self) -> str:
        """Return the fully qualified class path that identifies this encoder."""
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def _normalize_structure(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(pos.shape[0])
        if n == 0:
            raise ValueError("No residues found to encode.")

        pos_out = pos.astype(np.float32)

        if self.config.center:
            centroid = pos_out.mean(axis=0, keepdims=True).astype(np.float32)
            pos_out = pos_out - centroid
        else:
            centroid = np.zeros((1, 3), dtype=np.float32)

        if self.config.scale is not None:
            pos_out = (pos_out / float(self.config.scale)).astype(np.float32)

        return centroid, pos_out

    def _output_declarations(self, data: dict[str, np.ndarray]) -> OutputDeclarations:  # noqa: ARG002
        return self.output_declarations

    def _build_metadata(
        self,
        ctx: EncodingContext,
        data: dict[str, np.ndarray],
        *,
        parameters: dict[str, Any],
        extensions: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build prospective strict-valid metadata without activating it in encode()."""
        if ctx.source_sha256 is None:
            raise ValueError("prospective artifact metadata requires ctx.source_sha256")
        if ctx.input_format is None:
            raise ValueError("prospective artifact metadata requires ctx.input_format")

        representation_name = self.config.representation_name
        granularity = self.config.granularity

        return build_artifact_metadata(
            data=data,
            representation_name=representation_name,
            representation_family=self.representation_family,
            representation_type=self.representation.representation_type,
            granularity=granularity,
            coordinate_source=self.config.coordinate_source,
            encoder={
                "implementation": self.encoder_implementation,
                "version": self.version,
            },
            input_metadata=build_input_metadata(
                source=ctx.inp.source,
                source_sha256=ctx.source_sha256,
                input_format=ctx.input_format,
                model_index=ctx.inp.model_index,
                chain_ids=_chain_ids(ctx.inp),
            ),
            parameters=parameters,
            provenance=build_provenance(
                representation_name=representation_name,
                parameters=parameters,
                chain_ids=_chain_ids(ctx.inp),
            ),
            output_declarations=self._output_declarations(data),
            feature_keys=self.representation.feature_keys(data, ctx),
            counts=self.representation.counts(data),
            extensions=extensions,
        )

    def _matches_cache_identity(
        self,
        artifact: StructureArtifact,
        *,
        inp: StructureInput,
        source_sha256: str,
        parser_hint: str,
        parameters: dict[str, Any],
        config_fingerprint: str,
    ) -> bool:
        meta = artifact.meta
        encoder = meta.get("encoder")
        provenance = meta.get("provenance")
        input_meta = meta.get("input")
        if not isinstance(encoder, dict):
            return False
        if not isinstance(provenance, dict):
            return False
        if not isinstance(input_meta, dict):
            return False
        return (
            meta.get("representation_name") == self.config.representation_name
            and meta.get("representation_family") == self.representation_family
            and meta.get("representation_type") == self.representation.representation_type
            and meta.get("granularity") == self.config.granularity
            and meta.get("coordinate_source") == self.config.coordinate_source
            and meta.get("parameters") == parameters
            and encoder.get("implementation") == self.encoder_implementation
            and encoder.get("version") == self.version
            and provenance.get("config_fingerprint") == config_fingerprint
            and input_meta.get("source_sha256") == source_sha256
            and input_meta.get("model_index") == inp.model_index
            and input_meta.get("chain_ids") == list(_chain_ids(inp))
            and isinstance(input_meta.get("format"), str)
            and parser_hint_matches_format(parser_hint, input_meta["format"])
            and "model" not in meta
        )

    def _refresh_cache_hit_source(
        self,
        artifact: StructureArtifact,
        *,
        inp: StructureInput,
        source_sha256: str,
        extensions: dict[str, dict[str, Any]] | None,
    ) -> StructureArtifact:
        meta = deepcopy(artifact.meta)
        meta["input"] = build_input_metadata(
            source=inp.source,
            source_sha256=source_sha256,
            input_format=artifact.meta["input"]["format"],
            model_index=inp.model_index,
            chain_ids=_chain_ids(inp),
        )
        if extensions is not None:
            refreshed_extensions = deepcopy(meta.get("extensions", {}))
            refreshed_extensions.update(deepcopy(extensions))
            meta["extensions"] = refreshed_extensions
        refreshed = StructureArtifact(data=artifact.data, meta=meta)
        refreshed.validate()
        return refreshed

    def _validate_runtime_config_dependencies(self, parameters: dict[str, Any]) -> None:  # noqa: B027
        """Validate mutable resources referenced by canonical configuration."""

    def _artifact_extensions(
        self,
        parameters: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, dict[str, Any]] | None:
        """Return invocation provenance that does not participate in identity."""
        return None

    def encode(self, inp: StructureInput) -> StructureArtifact:
        """Encode one structure into a validated artifact, using the disk cache when enabled."""
        structure = structure_io.load_structure(Path(inp.source))
        inp = structure_io.resolve_structure_input(inp, structure)

        source_sha256 = sha256_file(Path(inp.source))
        invocation_parameters = self.config.to_dict()
        self._validate_runtime_config_dependencies(invocation_parameters)
        parameters = self.config.to_fingerprint_dict()
        extensions = self._artifact_extensions(invocation_parameters)
        config_fingerprint = compute_config_fingerprint(
            self.config.representation_name,
            parameters,
            chain_ids=_chain_ids(inp),
        )
        cache = None
        key = None
        if self.use_cache:
            cache = self.cache or DiskCache()
            parser_hint = parser_hint_from_source(inp.source)
            key = compute_cache_key(
                representation_name=self.config.representation_name,
                encoder_implementation=self.encoder_implementation,
                encoder_version=self.version,
                config_fingerprint=config_fingerprint,
                source_sha256=source_sha256,
                parser_hint=parser_hint,
                model_index=inp.model_index,
                chain_ids=_chain_ids(inp),
                model=None,
            )
            artifact = cache.get(key)
            if artifact is not None and self._matches_cache_identity(
                artifact,
                inp=inp,
                source_sha256=source_sha256,
                parser_hint=parser_hint,
                parameters=parameters,
                config_fingerprint=config_fingerprint,
            ):
                return self._refresh_cache_hit_source(
                    artifact,
                    inp=inp,
                    source_sha256=source_sha256,
                    extensions=extensions,
                )

        ctx = EncodingContext(inp=inp, source_sha256=source_sha256, structure=structure)
        self._prepare_data(ctx=ctx)
        _structure_validation(ctx=ctx)

        data = self._create_data(ctx=ctx)

        _check_output_values(data)

        meta = self._build_metadata(
            ctx,
            data,
            parameters=parameters,
            extensions=extensions,
        )
        output_names = [output["name"] for output in meta["outputs"]]
        data = {name: data[name] for name in output_names}
        artifact = StructureArtifact(data=data, meta=meta)
        artifact.validate()

        if cache is not None and key is not None:
            cache.set(key, artifact)

        return artifact

    def _prepare_data(self, ctx: EncodingContext) -> None:
        logger.info(
            "Loading structure",
            extra={
                "source": str(ctx.inp.source),
                "granularity": self.granularity,
            },
        )

        strc = self._load_structure(ctx)
        input_format = getattr(strc, "input_format", None)
        if not isinstance(input_format, str) or not input_format:
            raise exc.ExecutionError("Structure loader did not report a parsed input format.")
        if ctx.input_format is not None:
            raise exc.ExecutionError("Encoding context input format was already populated.")
        ctx.input_format = input_format
        ctx.raw_structure = strc

        centroid, pos_out = self._normalize_structure(pos=strc["pos"])
        ctx.centroid = centroid.astype(np.float32)
        ctx.pos = pos_out.astype(np.float32)

    @abstractmethod
    def _load_structure(self, ctx: EncodingContext) -> LoadedStructureData:
        """Load the structure arrays; implemented by the atom and residue base classes."""
        raise NotImplementedError

    @abstractmethod
    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        """Must be implemented in encoder class."""
        raise NotImplementedError
