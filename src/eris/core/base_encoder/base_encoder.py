from __future__ import annotations

from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, ClassVar, Tuple
from typing import TYPE_CHECKING

import json
import numpy as np
from pathlib import Path

from eris import exceptions as exc
from eris.core.cache import (
    DiskCache,
    compute_cache_key_v2,
    parser_hint_from_source,
    parser_hint_matches_format,
)
from eris.core.types import StructureArtifact, StructureInput
from eris.core.contracts import UniversalContract


if TYPE_CHECKING:
    from eris.core.cache import DiskCache
    from eris.core.contracts.base_contract import BaseContract
    from eris.core.contracts.granularity import GranularityContract
    from eris.core.representations.base_rep import BaseRepresentation


def _structure_validation(ctx: EncodingContext):
    strc = ctx.raw_structure
    n = int(strc["pos"].shape[0])

    if n < 1:
        logger.error(
            f"Structure: {ctx.inp.source} does not contain residues/atoms.",
            extra={"n_nodes": n, "input": ctx.inp.source},
        )
        raise exc.ExecutionError(f"Structure: {ctx.inp.source} does not contain residues/atoms.")

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


@dataclass
class BaseEncoder(ABC):
    """
    Base class for all encoders.
    Defines the contract for each subclass.
    """

    # Identity
    version: ClassVar[str] = "0.1.0"  # Extraer desde versión instalada / pyproject
    name: ClassVar[str]
    representation_family: ClassVar[str] = "explicit"
    granularity_contract: ClassVar[GranularityContract]
    representation: ClassVar[BaseRepresentation]
    output_declarations_v2: ClassVar[Any]

    # Instance attr
    use_cache: bool = True
    cache: Optional[DiskCache] = None
    center: bool = True  # center coordinates around centroid
    scale: Optional[float] = None  # if set, divide coords by this constant

    def __post_init__(self):
        mandatory = ["name", "granularity_contract", "representation"]
        cls = type(self)

        missing = [attr for attr in mandatory if not hasattr(cls, attr)]

        if missing:
            raise ValueError(f"Missing mandatory ClassVar in {cls.__name__}: {missing}")

    def contracts(self):
        return [UniversalContract(), self.granularity_contract] + self.representation.contracts()

    # Add to exclude granularity_contract and representation?
    def get_params(self) -> Dict[str, Any]:
        EXCLUDE = {"cache", "use_cache", "granularity_contract", "representation"}

        return {
            k: v for k, v in self.__dict__.items() if k not in EXCLUDE and not k.startswith("_")
        }

    def fingerprint(self) -> str:
        from eris.core.hashing import sha256_text

        payload = {
            "name": self.name,
            "version": self.version,
            "representation": self.representation.representation_type,
            "level": self.granularity_contract.level,
            "params": self.get_params(),
        }

        text = json.dumps(payload, sort_keys=True)
        return sha256_text(text)

    @property
    def encoder_implementation(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def _normalize_structure(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(pos.shape[0])
        if n == 0:
            raise ValueError("No residues found to encode.")

        pos_out = pos.astype(np.float32)

        if self.center and n > 0:
            centroid = pos_out.mean(axis=0, keepdims=True).astype(np.float32)
            pos_out = pos_out - centroid
        else:
            centroid = np.zeros((1, 3), dtype=np.float32)

        if self.scale is not None:
            s = float(self.scale)
            if s <= 0:
                raise ValueError("scale must be > 0")
            pos_out = (pos_out / s).astype(np.float32)

        return centroid, pos_out

    def _feature_keys_v2(self, data: dict[str, Any], ctx: EncodingContext):
        return self.representation.feature_keys_v2(data, ctx)

    def _counts_v2(self, data: dict[str, Any]):
        return self.representation.counts_v2(data)

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
        if representation_name != self.name:
            raise ValueError(
                "config representation_name does not match encoder name: "
                f"{representation_name!r} != {self.name!r}"
            )
        granularity = self.config.granularity
        if granularity != self.granularity_contract.level:
            raise ValueError(
                "config granularity does not match encoder contract: "
                f"{granularity!r} != {self.granularity_contract.level!r}"
            )

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
            output_declarations=self.output_declarations_v2,
            feature_keys=self._feature_keys_v2(data, ctx),
            counts=self._counts_v2(data),
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

        for contract in self.contracts():
            contract.validate(data)

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
