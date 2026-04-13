from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import json

if TYPE_CHECKING:
    from morpheo.core.types import StructureArtifact, StructureInput
    from typing import Any, Dict

class BaseEncoder(ABC):
    """
    Base class for all encoders.
    Defiens the contract for each subclass.
    """

    version: str = "0.1.0"
    name: str
    representation_type: str

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


    @abstractmethod
    def encode(self, inp: StructureInput) -> StructureArtifact:
        """
        Must be implemented in each encoder
        """
        pass


    def validate_output(self, artifact: StructureArtifact) -> None:
        artifact.validate(strict=True)