from __future__ import annotations

import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Tuple

import numpy as np

from eris import exceptions as exc
from eris.core.artifacts import StructureArtifact
from eris.core.cache import (
    DiskCache,
    compute_cache_key_v2,
    parser_hint_from_source,
    parser_hint_matches_format,
)
from eris.core.contracts import validate_encoder_output
from eris.core.types import StructureInput

if TYPE_CHECKING:
    from eris.core.config import CanonicalRepresentationConfig
    from eris.core.contracts.granularity import GranularityContract


def _structure_validation(ctx: EncodingContext):
    strc = ctx.raw_structure
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
    inp: StructureInput
    source_sha256: str | None = None
    input_format: str | None = None
    raw_structure: dict | None = None
    centroid: np.ndarray | None = None
    pos: np.ndarray | None = None
    runtime: dict[str, Any] = field(default_factory=dict)


class BaseEncoder(ABC):
    """
    Base class for all encoders.
    Defines the contract for each subclass.
    """

    # Identity
    version: ClassVar[str] = "0.1.0"
    name: ClassVar[str]
    representation_family: ClassVar[str] = "explicit"
    granularity_contract: ClassVar[GranularityContract]
    representation: ClassVar[Any]
    output_declarations_v2: ClassVar[Any]
    config_type: ClassVar[type[CanonicalRepresentationConfig]]

    def __init__(
        self,
        *,
        config: CanonicalRepresentationConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
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
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def _normalize_structure(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def _output_declarations_v2(self, data: dict[str, np.ndarray]):
        return self.output_declarations_v2

    def _build_metadata_v2(
        self,
        ctx: EncodingContext,
        data: dict[str, np.ndarray],
        *,
        parameters: dict[str, Any],
        extensions: dict[str, dict[str, Any]] | None = None,
    ):
        """Build prospective strict-valid v2 metadata without activating it in encode()."""
        from eris.core.artifact_schema_v2 import (
            build_artifact_metadata,
            build_input_metadata,
            build_provenance,
        )

        if ctx.source_sha256 is None:
            raise ValueError("prospective Schema v2 metadata requires ctx.source_sha256")
        if ctx.input_format is None:
            raise ValueError("prospective Schema v2 metadata requires ctx.input_format")

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
                chain_ids=ctx.inp.chain_id,
            ),
            parameters=parameters,
            provenance=build_provenance(
                representation_name=representation_name,
                parameters=parameters,
                chain_ids=ctx.inp.chain_id,
            ),
            output_declarations=self._output_declarations_v2(data),
            feature_keys=self.representation.feature_keys_v2(data, ctx),
            counts=self.representation.counts_v2(data),
            extensions=extensions,
        )

    def _matches_cache_identity_v2(
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
            type(meta.get("schema_version")) is int
            and meta["schema_version"] == 2
            and meta.get("representation_name") == self.config.representation_name
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
            and input_meta.get("chain_ids") == list(inp.chain_id)
            and isinstance(input_meta.get("format"), str)
            and parser_hint_matches_format(parser_hint, input_meta["format"])
            and "model" not in meta
        )

    def _refresh_cache_hit_source_v2(
        self,
        artifact: StructureArtifact,
        *,
        inp: StructureInput,
        source_sha256: str,
        extensions: dict[str, dict[str, Any]] | None,
    ) -> StructureArtifact:
        from eris.core.artifact_schema_v2 import build_input_metadata

        meta = deepcopy(artifact.meta)
        meta["input"] = build_input_metadata(
            source=inp.source,
            source_sha256=source_sha256,
            input_format=artifact.meta["input"]["format"],
            model_index=inp.model_index,
            chain_ids=inp.chain_id,
        )
        if extensions is not None:
            refreshed_extensions = deepcopy(meta.get("extensions", {}))
            refreshed_extensions.update(deepcopy(extensions))
            meta["extensions"] = refreshed_extensions
        refreshed = StructureArtifact(data=artifact.data, meta=meta)
        refreshed.validate(strict=True)
        return refreshed

    def _validate_runtime_config_dependencies(self, parameters: dict[str, Any]) -> None:
        """Validate mutable resources referenced by canonical configuration."""

    def _artifact_extensions(
        self,
        parameters: dict[str, Any],
    ) -> dict[str, dict[str, Any]] | None:
        """Return invocation provenance that does not participate in identity."""
        return None

    def encode(self, inp: StructureInput) -> StructureArtifact:
        from eris.core.artifact_schema_v2 import compute_config_fingerprint
        from eris.core.hashing import sha256_file
        from eris.core.io import resolve_structure_input

        inp = resolve_structure_input(inp)

        source_sha256 = sha256_file(Path(inp.source))
        invocation_parameters = self.config.to_dict()
        self._validate_runtime_config_dependencies(invocation_parameters)
        parameters = self.config.to_fingerprint_dict()
        extensions = self._artifact_extensions(invocation_parameters)
        config_fingerprint = compute_config_fingerprint(
            self.config.representation_name,
            parameters,
            chain_ids=inp.chain_id,
        )
        cache = None
        key = None
        if self.use_cache:
            cache = self.cache or DiskCache()
            parser_hint = parser_hint_from_source(inp.source)
            key = compute_cache_key_v2(
                representation_name=self.config.representation_name,
                encoder_implementation=self.encoder_implementation,
                encoder_version=self.version,
                config_fingerprint=config_fingerprint,
                source_sha256=source_sha256,
                parser_hint=parser_hint,
                model_index=inp.model_index,
                chain_ids=inp.chain_id,
                model=None,
            )
            artifact = cache.get_v2(key)
            if artifact is not None and self._matches_cache_identity_v2(
                artifact,
                inp=inp,
                source_sha256=source_sha256,
                parser_hint=parser_hint,
                parameters=parameters,
                config_fingerprint=config_fingerprint,
            ):
                self._validate_runtime_config_dependencies(invocation_parameters)
                return self._refresh_cache_hit_source_v2(
                    artifact,
                    inp=inp,
                    source_sha256=source_sha256,
                    extensions=extensions,
                )

        ctx = EncodingContext(inp=inp, source_sha256=source_sha256)
        self._prepare_data(ctx=ctx)
        _structure_validation(ctx=ctx)

        data = self._create_data(ctx=ctx)

        validate_encoder_output(
            data, self.representation.representation_type, self.granularity_contract
        )

        meta = self._build_metadata_v2(
            ctx,
            data,
            parameters=parameters,
            extensions=extensions,
        )
        output_names = [output["name"] for output in meta["outputs"]]
        data = {name: data[name] for name in output_names}
        artifact = StructureArtifact(data=data, meta=meta)
        artifact.validate(strict=True)

        if cache is not None and key is not None:
            cache.set_v2(key, artifact)

        return artifact

    def _prepare_data(self, ctx: EncodingContext):
        logger.info(
            "Loading structure",
            extra={
                "source": str(ctx.inp.source),
                "granularity": self.granularity_contract.level,
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
    def _load_structure(self, ctx: EncodingContext) -> Dict[str, Any]:
        """
        Implemented in base atom and base residue classes.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        """
        Must be implemented in encoder class.
        """
        raise NotImplementedError
