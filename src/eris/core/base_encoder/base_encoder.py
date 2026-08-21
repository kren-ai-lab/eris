from __future__ import annotations

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
from eris.core.encoder_metadata import check_params
from eris.core.cache import DiskCache, make_cache_key
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
                extra={"n_nodes": n,
                       "input": ctx.inp.source},
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
    #Identity 
    version: ClassVar[str] = "0.1.0" #Extraer desde versión instalada / pyproject
    name: ClassVar[str]
    granularity_contract: ClassVar[GranularityContract]
    representation: ClassVar[BaseRepresentation]

    #Instance attr
    use_cache: bool = True
    cache: Optional[DiskCache] = None
    center: bool = True           # center coordinates around centroid
    scale: Optional[float] = None # if set, divide coords by this constant


    def __post_init__(self):
        mandatory = ["name", "granularity_contract", "representation"]        
        cls = type(self)

        missing = [
            attr for attr in mandatory
            if not hasattr(cls, attr)
        ]

        if missing:
            raise ValueError(
                f"Missing mandatory ClassVar in {cls.__name__}: {missing}"
            )
        
    
    def contracts(self):
        return (
            [UniversalContract(), self.granularity_contract]
            + self.representation.contracts()
        )

    # Add to exclude granularity_contract and representation?
    def get_params(self) -> Dict[str, Any]:
        EXCLUDE = {"cache", "use_cache", "granularity_contract", "representation"}

        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in EXCLUDE and not k.startswith("_")
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
    

    def _check_cache(self, inp: StructureInput, source_sha256: str):
        cache = self.cache or DiskCache()
        key = make_cache_key(self.name, self.fingerprint(), source_sha256, inp.to_string())

        if self.use_cache:
            hit = cache.get(key)
            if hit is not None:
                data, meta = hit
                artifact = StructureArtifact(data=data, meta=meta)
                return key, cache, artifact
        
        return key, cache, None
    

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


    def _build_feature_keys(self, data, meta, ctx):
        return self.representation.feature_keys(data, meta, ctx)


    def encode(self, inp: StructureInput) -> StructureArtifact:
        from eris.core.hashing import sha256_file

        source_sha256 = sha256_file(Path(inp.source))
        key, cache, artifact = self._check_cache(inp, source_sha256)
        if artifact is not None:
            return artifact
        
        ctx = EncodingContext(inp=inp, source_sha256=source_sha256)
        self._prepare_data(ctx=ctx)
        _structure_validation(ctx=ctx)

        data = self._create_data(ctx=ctx)
        meta = self._base_metadata(ctx=ctx)
        meta = self._update_granularity_metadata(meta=meta, ctx=ctx)
        meta = self._update_params_metadata(meta=meta, ctx=ctx)
        meta = self._finalize_metadata(meta=meta)

        artifact: StructureArtifact = self._assemble_output(data=data, meta=meta, ctx=ctx)
        
        if self.use_cache:
            cache.set(key, data=artifact.data, meta=artifact.meta)

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


    @abstractmethod
    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        """
        Must be implemented in encoder class.
        """
        raise NotImplementedError
    

    def _update_granularity_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        """
        Must be implemented in base granularity (atom/residue) class.
        """
        raise NotImplementedError

    
    def _base_metadata(self, ctx: EncodingContext) -> Dict[str, Any]:
        inp = ctx.inp
        src = inp.source
        fmt = Path(src).suffix.lstrip(".").lower() or None

        return {
            "encoder_name": self.name,
            "encoder_version": self.version,
            "type": self.representation.representation_type,
            "level": self.granularity_contract.level,
            "input": {
                "source": src,
                "chain_id": inp.chain_id,
                "model_index": inp.model_index,
                "format": fmt,
            },
            "params": {
                "center": bool(self.center),
                "scale": None if self.scale is None else float(self.scale),
            },
        }
    
   
    def _finalize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        if "params" in meta:
            meta["params"] = check_params(**meta["params"])
        return meta
    

    def _assemble_output(
        self,
        data: Dict[str, Any],
        meta: Dict[str, Any],
        ctx: EncodingContext,
    ) -> StructureArtifact:
        for contract in self.contracts():
            contract.validate(data)

        meta["feature_keys"] = self._build_feature_keys(data, meta, ctx)

        meta["encoder_fingerprint"] = self.fingerprint()

        return StructureArtifact(data=data, meta=meta)


    def validate_output(self, artifact: StructureArtifact) -> None:
        artifact.validate(strict=True)
