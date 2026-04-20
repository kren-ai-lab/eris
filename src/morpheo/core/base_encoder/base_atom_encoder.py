from morpheo.core.base_encoder.base_encoder import BaseEncoder

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

@dataclass
class BaseAtomEncoder(BaseEncoder):
    pass