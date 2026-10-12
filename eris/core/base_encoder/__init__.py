"""Base classes for explicit-representation encoders."""

from eris.core.base_encoder.base_atom_encoder import BaseAtomEncoder
from eris.core.base_encoder.base_encoder import EncodingContext
from eris.core.base_encoder.base_residue_encoder import BaseResidueEncoder

__all__ = ["BaseAtomEncoder", "BaseResidueEncoder", "EncodingContext"]
