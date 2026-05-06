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

from morpheo import exceptions as exc
from morpheo.core.encoder_metadata import check_params
from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.types import StructureArtifact, StructureInput
from morpheo.core.contracts import UniversalContract


if TYPE_CHECKING:
    from morpheo.core.cache import DiskCache
    from morpheo.core.contracts.base_contract import BaseContract
    from morpheo.core.contracts.granularity import GranularityContract
    from morpheo.core.representations.base_rep import BaseRepresentation


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
        EXCLUDE = {"cache", "granularity_contract", "representation"}

        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in EXCLUDE and not k.startswith("_")
        }


    def fingerprint(self) -> str:
        from morpheo.core.hashing import sha256_text

        payload = {
            "name": self.name,
            "version": self.version,
            "representation": self.representation.representation_type,
            #"family": self.representation.family,
            "level": self.granularity_contract.level,
            "params": self.get_params(),
        }

        text = json.dumps(payload, sort_keys=True)
        return sha256_text(text)
    

    def _check_cache(self, inp: StructureInput):
        from morpheo.core.hashing import sha256_file

        cache = self.cache or DiskCache()
        file_hash = sha256_file(Path(inp.source))
        key = make_cache_key(self.name, self.fingerprint(), file_hash, inp.to_string())

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


    def _build_feature_keys(self, data, meta):
        return self.representation.feature_keys(data, meta)


    def encode(self, inp: StructureInput) -> StructureArtifact:
        key, cache, artifact = self._check_cache(inp)
        if artifact is not None:
            return artifact
        
        ctx = EncodingContext(inp=inp)
        self._prepare_data(ctx=ctx)
        _structure_validation(ctx=ctx)

        data = self._create_data(ctx=ctx)
        meta = self._base_metadata(ctx=ctx)
        meta = self._update_params_metadata(meta=meta, ctx=ctx)
        meta = self._finalize_metadata(meta=meta)

        artifact: StructureArtifact = self._assemble_output(data=data, meta=meta)
        
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
        pass


    @abstractmethod
    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        """
        Must be implemented in encoder class.
        """
        pass


    @abstractmethod
    def _update_params_metadata(self, ctx: EncodingContext) -> Dict[str, Any]:
        """
        Must be implemented in encoder class.
        """
        pass

    
    def _base_metadata(self, ctx: EncodingContext) -> Dict[str, Any]:
        inp = ctx.inp
        src = inp.source
        fmt = Path(src).suffix.lstrip(".").lower() or None

        return {
            "encoder_name": self.name,
            "encoder_version": self.version,
            "type": self.representation.representation_type,
            #"family": self.family,
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
        #ctx: EncodingContext,
    ) -> StructureArtifact:
        for contract in self.contracts():
            contract.validate(data)

        meta["feature_keys"] = self._build_feature_keys(data, meta)

        meta["encoder_fingerprint"] = self.fingerprint()

        return StructureArtifact(data=data, meta=meta)


    def validate_output(self, artifact: StructureArtifact) -> None:
        artifact.validate(strict=True)
