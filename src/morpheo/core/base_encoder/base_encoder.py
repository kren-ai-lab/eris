from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

import json
from pathlib import Path

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.types import StructureArtifact, StructureInput


if TYPE_CHECKING:
    from morpheo.core.cache import DiskCache
    from typing import Any, Dict


@dataclass
class BaseEncoder(ABC):
    """
    Base class for all encoders.
    Defiens the contract for each subclass.
    """

    version: ClassVar[str] = "0.1.0"
    name: ClassVar[str]
    representation_type: ClassVar[str] # old type
    family: ClassVar[str]
    level: ClassVar[str]

    use_cache: bool = True
    cache: Optional[DiskCache] = None


    # def __post_init__(self):
    #     mandatory = ["name", "representation_type", "family", "level"]        
    #     cls = type(self)

    #     missing = [
    #         attr for attr in mandatory
    #         if not hasattr(cls, attr)
    #     ]

    #     if missing:
    #         raise ValueError(
    #             f"Missing mandatory ClassVar in {cls.__name__}: {missing}"
    #         )


    def get_params(self) -> Dict[str, Any]:
        EXCLUDE = {"cache"}

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
    

    def encode(self, inp: StructureInput) -> StructureArtifact:
        key, cache, artifact = self._check_cache(inp)
        if artifact is not None:
            return artifact

        data =  self._create_data(inp=inp)
        meta = self._create_metadata(inp=inp)

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(data=data, meta=meta)
    

    @abstractmethod
    def _create_data(self, inp: StructureInput) -> Dict[str, Any]:
        """
        Must be implemented in base class.
        """
        pass


    @abstractmethod
    def _create_metadata(self, inp: StructureInput) -> Dict[str, Any]:
        """
        Must be implemented in base class.
        """
        pass

    
    def _base_metadata(self, inp: StructureInput) -> Dict[str, Any]:
        src = inp.source
        fmt = Path(src).suffix.lstrip(".").lower() or None

        return {
            "encoder_name": self.name,
            "encoder_version": self.version,
            "encoder_fingerprint": self.fingerprint(),
            "type": self.representation_type,
            "family": self.family,
            "level": self.level,
            "input": {
                "source": src,
                "chain_id": inp.chain_id,
                "model_index": inp.model_index,
                "format": fmt,
            },
        }
    

    def validate_output(self, artifact: StructureArtifact) -> None:
        artifact.validate(strict=True)
