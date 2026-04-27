from typing import Any, Dict, ClassVar

from morpheo.core.contracts.base_contract import GranularityContract

# not implemented
class ResidueContract(GranularityContract):
    """
    """
    level: ClassVar[str] = "residue"

    def validate(self, data: Dict[str, Any]):
        pass

# not implemented
class AtomContract(GranularityContract):
    """
    """
    level: ClassVar[str] = "residue"

    def validate(self, data: Dict[str, Any]):
        pass
