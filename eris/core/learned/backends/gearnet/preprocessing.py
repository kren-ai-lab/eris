"""Policy-neutral structural transport for the frozen GearNet-Edge backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gemmi
import numpy as np

from eris.core.io import _parsed_input_format, _select_atom_altloc, load_structure, select_chains
from eris.core.learned.backends.gearnet.metadata import GEARNET_EDGE_CANONICAL_RESIDUES
from eris.core.residue_identity import format_residue_id
from eris.exceptions import ExecutionError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eris.core.types import StructureInput

_CANONICAL_AMINO_ACIDS = frozenset(GEARNET_EDGE_CANONICAL_RESIDUES)


class GearNetPreprocessingError(ExecutionError):
    """Raised when policy-neutral GearNet preprocessing cannot preserve its contract."""


@dataclass(frozen=True, slots=True)
class BiologicalResidueIdentity:
    """One input residue identity, before any model-specific biological selection."""

    index: int
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    node_id: str


@dataclass(frozen=True, slots=True)
class BiologicalIdentityTable:
    """Ordered biological identities used as the authority for final model-node mapping."""

    records: tuple[BiologicalResidueIdentity, ...]

    def __post_init__(self) -> None:
        """Validate that identity indexes are contiguous and ordered."""
        if any(type(record.index) is not int or record.index < 0 for record in self.records):
            raise GearNetPreprocessingError("biological identity indexes must be non-negative integers")
        indexes = [record.index for record in self.records]
        if indexes != list(range(len(self.records))):
            raise GearNetPreprocessingError(
                "biological identity indexes must be contiguous and preserve input order"
            )
        node_ids = [record.node_id for record in self.records]
        if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
            raise GearNetPreprocessingError("biological identity node_ids must be non-empty strings")
        if len(node_ids) != len(set(node_ids)):
            raise GearNetPreprocessingError("biological identity node_ids must be unique")

    def node_ids_for(self, carrier: Sequence[int]) -> tuple[str, ...]:
        """Return the unique node IDs for the carrier indexes."""
        indexes = _validate_carrier(carrier, table_size=len(self.records))
        node_ids = tuple(self.records[index].node_id for index in indexes)
        if len(node_ids) != len(set(node_ids)):
            raise GearNetPreprocessingError("final model node_ids must be unique")
        return node_ids


@dataclass(frozen=True, slots=True)
class SelectedBiologicalStructure:
    """Gemmi selection plus policy-neutral identities in source order."""

    structure: gemmi.Structure
    chain: gemmi.Chain
    input_format: str
    model_index: int
    biological_table: BiologicalIdentityTable


@dataclass(frozen=True, slots=True)
class TransportNodeRecord:
    """One already-selected model node for native graph construction."""

    transport_index: int
    biological_index: int
    residue_name: str
    coordinate: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GearNetPreparedInput:
    """Internal handoff from structural preprocessing to future GearNet inference."""

    graph: Any
    carrier: np.ndarray
    biological_table: BiologicalIdentityTable
    node_ids: np.ndarray


def build_biological_identity_table(chain: gemmi.Chain) -> BiologicalIdentityTable:
    """Record every residue in a selected chain without applying biological policy."""
    records = []
    for index, residue in enumerate(chain):
        insertion_code = (residue.seqid.icode or "").strip()
        records.append(
            BiologicalResidueIdentity(
                index=index,
                chain_id=chain.name,
                residue_number=cast("int", residue.seqid.num),
                insertion_code=insertion_code,
                residue_name=residue.name,
                node_id=format_residue_id(
                    chain_id=chain.name,
                    residue_number=cast("int", residue.seqid.num),
                    insertion_code=insertion_code,
                    residue_name=residue.name,
                ),
            )
        )
    return BiologicalIdentityTable(records=tuple(records))


def load_selected_biological_structure(
    structure_input: StructureInput,
) -> SelectedBiologicalStructure:
    """Load and select one model/chain while deliberately applying no residue policy."""
    structure = load_structure(Path(structure_input.source))
    chains = select_chains(
        structure, cast("tuple[str, ...] | None", structure_input.chain_id), structure_input.model_index
    )
    if len(chains) != 1:
        raise GearNetPreprocessingError("GearNet v0.1 requires exactly one selected chain")
    chain = chains[0]
    return SelectedBiologicalStructure(
        structure=structure,
        chain=chain,
        input_format=_parsed_input_format(structure),
        model_index=structure_input.model_index,
        biological_table=build_biological_identity_table(chain),
    )


def _is_gearnet_canonical_residue(residue: gemmi.Residue) -> bool:
    residue_name = (residue.name or "").strip()
    residue_info = gemmi.find_tabulated_residue(residue_name)
    return residue.het_flag == "A" and residue_info.is_amino_acid() and residue_name in _CANONICAL_AMINO_ACIDS


def select_gearnet_transport_records(
    selected: SelectedBiologicalStructure,
    *,
    altloc: str = "first",
) -> tuple[TransportNodeRecord, ...]:
    """Apply the conservative GearNet v0.1 residue and CA selection policy."""
    if not isinstance(selected, SelectedBiologicalStructure):
        raise TypeError("selected must be a SelectedBiologicalStructure")
    identities = selected.biological_table.records
    if len(selected.chain) != len(identities):
        raise GearNetPreprocessingError("biological identity table must align with the selected chain")

    records: list[TransportNodeRecord] = []
    for residue, identity in zip(selected.chain, identities, strict=True):
        if not _is_gearnet_canonical_residue(residue):
            continue

        ca_atom = _select_atom_altloc(residue["CA"], altloc) if "CA" in residue else None
        if ca_atom is None:
            raise GearNetPreprocessingError(
                "GearNet v0.1 preprocessing requires CA for every canonical residue; "
                f"missing CA at {identity.node_id}"
            )
        coordinate = (
            float(ca_atom.pos.x),
            float(ca_atom.pos.y),
            float(ca_atom.pos.z),
        )
        if not all(math.isfinite(value) for value in coordinate):
            raise GearNetPreprocessingError(
                "GearNet v0.1 preprocessing requires a finite CA coordinate for every "
                f"canonical residue; unusable CA at {identity.node_id}"
            )
        records.append(
            TransportNodeRecord(
                transport_index=len(records),
                biological_index=identity.index,
                residue_name=identity.residue_name,
                coordinate=coordinate,
            )
        )

    if not records:
        raise GearNetPreprocessingError("No valid canonical residues remain for GearNet v0.1 preprocessing")
    return tuple(records)


def _validate_transport_records(
    records: Sequence[TransportNodeRecord],
    *,
    table_size: int | None = None,
) -> tuple[TransportNodeRecord, ...]:
    normalized = tuple(records)
    if not normalized:
        raise GearNetPreprocessingError("transport records must not be empty")

    if any(type(record.transport_index) is not int or record.transport_index < 0 for record in normalized):
        raise GearNetPreprocessingError("transport indexes must be non-negative integers")
    if any(type(record.biological_index) is not int or record.biological_index < 0 for record in normalized):
        raise GearNetPreprocessingError("biological indexes must be non-negative integers")

    transport_indexes = [record.transport_index for record in normalized]
    if len(transport_indexes) != len(set(transport_indexes)):
        raise GearNetPreprocessingError("transport indexes must be unique")
    if transport_indexes != list(range(len(normalized))):
        raise GearNetPreprocessingError("transport indexes must be contiguous and preserve collection order")

    biological_indexes = [record.biological_index for record in normalized]
    if len(biological_indexes) != len(set(biological_indexes)):
        raise GearNetPreprocessingError("biological carrier indexes must be unique")
    if table_size is not None and any(
        type(index) is not int or index < 0 or index >= table_size for index in biological_indexes
    ):
        raise GearNetPreprocessingError("biological carrier index is outside the identity table")

    for record in normalized:
        if not isinstance(record.residue_name, str):
            raise GearNetPreprocessingError("transport residue names must be strings")
        residue_name = record.residue_name.strip()
        if (
            not residue_name
            or len(residue_name) > 3
            or not residue_name.isascii()
            or not residue_name.isalnum()
        ):
            raise GearNetPreprocessingError(
                "transport residue names must be 1-3 ASCII alphanumeric characters"
            )
        try:
            coordinate = tuple(record.coordinate)
        except TypeError as error:
            raise GearNetPreprocessingError("transport coordinates must be three finite numbers") from error
        if len(coordinate) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
            for value in coordinate
        ):
            raise GearNetPreprocessingError("transport coordinates must be three finite numbers")

    return normalized


def _validate_carrier(carrier: Sequence[int], *, table_size: int) -> tuple[int, ...]:
    indexes = tuple(carrier)
    if len(indexes) != len(set(indexes)):
        raise GearNetPreprocessingError("model-node carrier indexes must be unique")
    if any(type(index) is not int or index < 0 or index >= table_size for index in indexes):
        raise GearNetPreprocessingError("model-node carrier index is outside the identity table")
    return indexes


def construct_native_gearnet_input(
    records: Sequence[TransportNodeRecord],
    biological_table: BiologicalIdentityTable,
) -> GearNetPreparedInput:
    """Build the native graph while preserving the existing biological carrier contract."""
    normalized = _validate_transport_records(records, table_size=len(biological_table.records))
    from eris.core.learned.backends.gearnet.native_graph import construct_native_gearnet_graph  # noqa: PLC0415

    graph = construct_native_gearnet_graph(normalized)
    carrier = np.asarray([record.biological_index for record in normalized], dtype=np.int64)
    carrier_tuple = _validate_carrier(carrier.tolist(), table_size=len(biological_table.records))
    node_ids = np.asarray(biological_table.node_ids_for(carrier_tuple), dtype=np.str_)
    return GearNetPreparedInput(
        graph=graph,
        carrier=carrier,
        biological_table=biological_table,
        node_ids=node_ids,
    )


__all__ = [
    "BiologicalIdentityTable",
    "BiologicalResidueIdentity",
    "GearNetPreparedInput",
    "GearNetPreprocessingError",
    "SelectedBiologicalStructure",
    "TransportNodeRecord",
    "build_biological_identity_table",
    "construct_native_gearnet_input",
    "load_selected_biological_structure",
    "select_gearnet_transport_records",
]
