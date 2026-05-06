from abc import ABC, abstractmethod
from typing import Any, Dict, ClassVar


class BaseContract(ABC):
    @abstractmethod
    def validate(self, data: Dict[str, Any]):
        ...

class EncoderContract(BaseContract):
    contract_name: ClassVar[str]