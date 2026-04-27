import numpy as np

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

from morpheo.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from morpheo.core.contracts.base_contract import GranularityContract
from morpheo.core.contracts import AtomContract
from morpheo.core.io import load_and_extract_atoms

@dataclass
class BaseAtomEncoder(BaseEncoder):
    granularity_contract: ClassVar[GranularityContract] = AtomContract()

    def _prepare_data(self, ctx: EncodingContext):
        strc = load_and_extract_atoms(ctx.inp)
        ctx.raw_structure = strc

        centroid, pos_out = self._normalize_structure(pos=strc["pos"])
        ctx.centroid = centroid.astype(np.float32)
        ctx.pos = pos_out.astype(np.float32)