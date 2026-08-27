"""Policy-neutral structural transport for the frozen GearNet-Edge backend."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from eris.core.io import _parsed_input_format, _select_atom_altloc, load_structure, select_chain
from eris.core.residue_identity import format_residue_id
from eris.core.types import StructureInput
from eris.exceptions import ExecutionError

_SYNTHETIC_CHAIN_ID = "A"
_FIRST_SYNTHETIC_RESIDUE_NUMBER = 2
_MAX_PDB_RESIDUE_NUMBER = 9999
_NODE_FEATURE_WIDTH = 21
_EDGE_FEATURE_WIDTH = 59
_NUM_RELATION = 7
_CANONICAL_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


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
    model_coordinate: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class BiologicalIdentityTable:
    """Ordered biological identities used as the authority for final model-node mapping."""

    records: tuple[BiologicalResidueIdentity, ...]

    def __post_init__(self) -> None:
        if any(type(record.index) is not int or record.index < 0 for record in self.records):
            raise GearNetPreprocessingError(
                "biological identity indexes must be non-negative integers"
            )
        indexes = [record.index for record in self.records]
        if indexes != list(range(len(self.records))):
            raise GearNetPreprocessingError(
                "biological identity indexes must be contiguous and preserve input order"
            )
        node_ids = [record.node_id for record in self.records]
        if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
            raise GearNetPreprocessingError(
                "biological identity node_ids must be non-empty strings"
            )
        if len(node_ids) != len(set(node_ids)):
            raise GearNetPreprocessingError("biological identity node_ids must be unique")

    def node_ids_for(self, carrier: Sequence[int]) -> tuple[str, ...]:
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
    """One already-selected model node encoded through synthetic PDB identity."""

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
                residue_number=residue.seqid.num,
                insertion_code=insertion_code,
                residue_name=residue.name,
                node_id=format_residue_id(
                    chain_id=chain.name,
                    residue_number=residue.seqid.num,
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
    chain = select_chain(structure, structure_input.chain_id, structure_input.model_index)
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
    return (
        residue.het_flag == "A"
        and residue_info.is_amino_acid()
        and residue_name in _CANONICAL_AMINO_ACIDS
    )


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
        raise GearNetPreprocessingError(
            "biological identity table must align with the selected chain"
        )

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
        raise GearNetPreprocessingError(
            "No valid canonical residues remain for GearNet v0.1 preprocessing"
        )
    return tuple(records)


def _validate_transport_records(
    records: Sequence[TransportNodeRecord],
    *,
    table_size: int | None = None,
) -> tuple[TransportNodeRecord, ...]:
    normalized = tuple(records)
    if not normalized:
        raise GearNetPreprocessingError("transport records must not be empty")
    if len(normalized) > _MAX_PDB_RESIDUE_NUMBER - _FIRST_SYNTHETIC_RESIDUE_NUMBER + 1:
        raise GearNetPreprocessingError("synthetic PDB residue numbering exceeds 9999")

    if any(
        type(record.transport_index) is not int or record.transport_index < 0
        for record in normalized
    ):
        raise GearNetPreprocessingError("transport indexes must be non-negative integers")
    if any(
        type(record.biological_index) is not int or record.biological_index < 0
        for record in normalized
    ):
        raise GearNetPreprocessingError("biological indexes must be non-negative integers")

    transport_indexes = [record.transport_index for record in normalized]
    if len(transport_indexes) != len(set(transport_indexes)):
        raise GearNetPreprocessingError("transport indexes must be unique")
    if transport_indexes != list(range(len(normalized))):
        raise GearNetPreprocessingError(
            "transport indexes must be contiguous and preserve collection order"
        )

    biological_indexes = [record.biological_index for record in normalized]
    if len(biological_indexes) != len(set(biological_indexes)):
        raise GearNetPreprocessingError("biological carrier indexes must be unique")
    if table_size is not None and any(
        type(index) is not int or index < 0 or index >= table_size for index in biological_indexes
    ):
        raise GearNetPreprocessingError("biological carrier index is outside the identity table")

    synthetic_ids: set[tuple[str, int, str]] = set()
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
            raise GearNetPreprocessingError(
                "transport coordinates must be three finite numbers"
            ) from error
        if len(coordinate) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in coordinate
        ):
            raise GearNetPreprocessingError("transport coordinates must be three finite numbers")

        synthetic_number = _FIRST_SYNTHETIC_RESIDUE_NUMBER + record.transport_index
        synthetic_id = (_SYNTHETIC_CHAIN_ID, synthetic_number, "")
        if synthetic_id in synthetic_ids:
            raise GearNetPreprocessingError("synthetic residue identifiers must be unique")
        synthetic_ids.add(synthetic_id)
    return normalized


def _pdb_atom_line(serial: int, record: TransportNodeRecord) -> str:
    residue_number = _FIRST_SYNTHETIC_RESIDUE_NUMBER + record.transport_index
    x, y, z = (float(value) for value in record.coordinate)
    residue_name = record.residue_name.strip().upper()
    return (
        f"ATOM  {serial:5d}  CA  {residue_name:>3s} {_SYNTHETIC_CHAIN_ID}"
        f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{0.0:6.2f}           C  "
    )


def build_synthetic_pdb(records: Sequence[TransportNodeRecord]) -> str:
    """Build deterministic single-chain CA-only PDB transport text."""

    normalized = _validate_transport_records(records)
    lines = [_pdb_atom_line(index + 1, record) for index, record in enumerate(normalized)]
    lines.extend(("TER", "END"))
    return "\n".join(lines) + "\n"


@contextmanager
def temporary_synthetic_pdb(records: Sequence[TransportNodeRecord]) -> Iterator[Path]:
    """Yield a Windows-compatible temporary PDB and remove it after use."""

    pdb_text = build_synthetic_pdb(records)
    with tempfile.TemporaryDirectory(prefix="eris-gearnet-") as directory:
        path = Path(directory) / "transport.pdb"
        path.write_text(pdb_text, encoding="ascii", newline="\n")
        yield path


def _validate_carrier(carrier: Sequence[int], *, table_size: int) -> tuple[int, ...]:
    indexes = tuple(carrier)
    if len(indexes) != len(set(indexes)):
        raise GearNetPreprocessingError("model-node carrier indexes must be unique")
    if any(type(index) is not int or index < 0 or index >= table_size for index in indexes):
        raise GearNetPreprocessingError("model-node carrier index is outside the identity table")
    return indexes


def _scalar_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def construct_gearnet_input(
    records: Sequence[TransportNodeRecord],
    biological_table: BiologicalIdentityTable,
) -> GearNetPreparedInput:
    """Materialize the frozen TorchDrug graph from already-selected model-node records."""

    normalized = _validate_transport_records(records, table_size=len(biological_table.records))

    # Optional backend imports remain behind the M5.2 availability boundary.
    import torch
    from torchdrug import data, layers, transforms
    from torchdrug.layers import geometry

    with temporary_synthetic_pdb(normalized) as pdb_path:
        protein = data.Protein.from_pdb(
            str(pdb_path),
            atom_feature=None,
            bond_feature=None,
            residue_feature="default",
        )
    if protein is None:
        raise GearNetPreprocessingError("TorchDrug did not construct a Protein")

    expected_numbers = [
        _FIRST_SYNTHETIC_RESIDUE_NUMBER + record.transport_index for record in normalized
    ]
    parsed_numbers = protein.residue_number.tolist()
    if parsed_numbers != expected_numbers:
        raise GearNetPreprocessingError(
            "TorchDrug synthetic residue identities do not match the transport records"
        )

    carrier_values = [record.biological_index for record in normalized]
    with protein.residue():
        protein.eris_residue_index = torch.tensor(
            carrier_values,
            dtype=torch.long,
            device=protein.device,
        )

    protein = transforms.ProteinView(view="residue")({"graph": protein})["graph"]
    packed = data.Protein.pack([protein])
    graph_construction = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SequentialEdge(max_distance=2),
            geometry.SpatialEdge(
                radius=10.0,
                min_distance=5,
                max_distance=None,
                max_num_neighbors=32,
            ),
            geometry.KNNEdge(k=10, min_distance=5, max_distance=None),
        ],
        edge_feature="gearnet",
    )
    graph = graph_construction(packed)

    if _scalar_int(graph.batch_size) != 1:
        raise GearNetPreprocessingError("GearNet preprocessing must produce exactly one graph")
    num_node = _scalar_int(graph.num_node)
    if num_node != _scalar_int(graph.num_residue):
        raise GearNetPreprocessingError("GearNet graph must have one node per retained residue")
    carrier = graph.eris_residue_index.detach().cpu().numpy().astype(np.int64, copy=True)
    if carrier.ndim != 1 or carrier.shape[0] != num_node:
        raise GearNetPreprocessingError("model-node carrier must align with graph nodes")
    carrier_tuple = _validate_carrier(carrier.tolist(), table_size=len(biological_table.records))
    if graph.node_feature.ndim != 2 or graph.node_feature.shape[1] != _NODE_FEATURE_WIDTH:
        raise GearNetPreprocessingError("GearNet node_feature width must be 21")
    if graph.edge_feature.ndim != 2 or graph.edge_feature.shape[1] != _EDGE_FEATURE_WIDTH:
        raise GearNetPreprocessingError("GearNet edge_feature width must be 59")
    if _scalar_int(graph.num_relation) != _NUM_RELATION:
        raise GearNetPreprocessingError("GearNet graph must declare 7 relations")

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
    "build_synthetic_pdb",
    "construct_gearnet_input",
    "load_selected_biological_structure",
    "select_gearnet_transport_records",
    "temporary_synthetic_pdb",
]
