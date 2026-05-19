import numpy as np

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

from eris.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from eris.core.contracts.granularity import GranularityContract
from eris.core.contracts import ResidueContract
from eris.core.io import load_and_extract


@dataclass
class BaseResidueEncoder(BaseEncoder):
    granularity_contract: ClassVar[GranularityContract] = ResidueContract()

    def _load_structure(self, ctx: EncodingContext):
        strc = load_and_extract(ctx.inp)
        return strc