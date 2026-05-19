from abc import ABC, abstractmethod
from typing import Any, Dict, ClassVar


class BaseRepresentation(ABC):
    representation_type: ClassVar[str]

    @abstractmethod
    def contracts(self):
        pass


    @abstractmethod
    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        pass


