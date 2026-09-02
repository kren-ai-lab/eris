from abc import ABC, abstractmethod
from typing import ClassVar


class BaseRepresentation(ABC):
    representation_type: ClassVar[str]

    @abstractmethod
    def contracts(self):
        pass
