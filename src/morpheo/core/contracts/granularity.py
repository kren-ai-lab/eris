from typing import Any, Dict, ClassVar

from morpheo.exceptions import ExecutionError
from morpheo.core.contracts.base_contract import BaseContract

class GranularityContract(BaseContract):
    level: ClassVar[str]
    REQUIRED_FIELDS: ClassVar[set[str]] 

    def validate(self, data: Dict[str, Any]) -> None:
        # Required fields must exist
        missing = self.REQUIRED_FIELDS - data.keys()
        if missing:
            raise ExecutionError(
                f"GranularityContract violation: missing required fields: {sorted(missing)}"
            )

        # All required fields must be sequence-like
        lengths = {}
        for field in self.REQUIRED_FIELDS:
            value = data[field]

            if not hasattr(value, "__len__"):
                raise TypeError(
                    f"GranularityContract violation: field '{field}' is not sequence-like"
                )

            lengths[field] = len(value)

        # All fields must agree on cardinality
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ExecutionError(
                "GranularityContract violation: inconsistent atom/residue field lengths: "
                + ", ".join(f"{k}={v}" for k, v in lengths.items())
            )

        # Must represent at least one atom
        n_nodes = unique_lengths.pop()
        if n_nodes < 1:
            raise ExecutionError("GranularityContract violation: no atoms/residues present")



class ResidueContract(GranularityContract):
    level: ClassVar[str] = "residue"
    REQUIRED_FIELDS: ClassVar[set[str]] = {
        "aa_index",
        "plddt",
        "bfactor_ca"
    }



class AtomContract(GranularityContract):
    level: ClassVar[str] = "atom"
    REQUIRED_FIELDS: ClassVar[set[str]] = {
        "elem_index",
        "atom_role",
        "atom_name",
        "bfactor",
        "occ",
        "seq_pos_residue",
        "aa_index_residue",
    }


