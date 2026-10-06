from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

OutputDeclaration = Mapping[str, str]
OutputDeclarations = tuple[OutputDeclaration, ...]


def _output(
    name: str,
    role: str,
    *,
    granularity: str | None = None,
    layout: str | None = None,
) -> OutputDeclaration:
    declaration = {"name": name, "role": role}
    if granularity is not None:
        declaration["granularity"] = granularity
    if layout is not None:
        declaration["layout"] = layout
    return MappingProxyType(declaration)


_CENTROID_OUTPUT = _output("centroid", "centroid", granularity="structure")
_GRAPH_OUTPUTS: OutputDeclarations = (
    _output("edge_index", "edge_index", layout="2E"),
    _output("edge_dist", "edge_distance", layout="E"),
    _output("edge_attr", "edge_features", layout="EF"),
    _output("graph_attr", "graph_features", granularity="structure", layout="F"),
)

_ATOM_AUX_OUTPUTS: OutputDeclarations = (
    _output("chain_id", "chain_id", granularity="atom", layout="N"),
    _output("elem_index", "element_index", granularity="atom", layout="N"),
    _output("atom_role", "atom_role", granularity="atom", layout="N"),
    _output("bfactor", "bfactor", granularity="atom", layout="N"),
    _output("occ", "occupancy", granularity="atom", layout="N"),
    _output("residue_ids", "residue_ids", granularity="atom", layout="N"),
    _output("resname", "residue_names", granularity="atom", layout="N"),
    _output("atom_name", "atom_names", granularity="atom", layout="N"),
    _output(
        "seq_pos_residue",
        "residue_sequence_index",
        granularity="atom",
        layout="N",
    ),
    _output("aa_index_residue", "amino_acid_index", granularity="atom", layout="N"),
)

_RESIDUE_AUX_OUTPUTS: OutputDeclarations = (
    _output("pos", "coordinates", granularity="residue", layout="N3"),
    _CENTROID_OUTPUT,
    _output("residue_ids", "node_ids", granularity="residue", layout="N"),
    _output("chain_id", "chain_id", granularity="residue", layout="N"),
    _output("omitted_residue_ids", "omitted_residue_ids", granularity="residue", layout="N"),
    _output("resname", "residue_names", granularity="residue", layout="N"),
    _output("aa_index", "amino_acid_index", granularity="residue", layout="N"),
    _output("plddt", "confidence", granularity="residue", layout="N"),
    _output("bfactor_ca", "bfactor", granularity="residue", layout="N"),
)

ATOM_POINT_CLOUD_OUTPUTS: OutputDeclarations = (
    _output("pos", "coordinates", granularity="atom", layout="N3"),
    _CENTROID_OUTPUT,
    _output("node_attr", "node_features", granularity="atom", layout="NF"),
    *_ATOM_AUX_OUTPUTS,
)

RESIDUE_POINT_CLOUD_OUTPUTS: OutputDeclarations = (
    _RESIDUE_AUX_OUTPUTS[0],
    _CENTROID_OUTPUT,
    _output("node_attr", "node_features", granularity="residue", layout="NF"),
    *_RESIDUE_AUX_OUTPUTS[2:],
)

ATOM_GRAPH_OUTPUTS: OutputDeclarations = (
    *ATOM_POINT_CLOUD_OUTPUTS[:3],
    *_GRAPH_OUTPUTS,
    *ATOM_POINT_CLOUD_OUTPUTS[3:],
)

RESIDUE_GRAPH_OUTPUTS: OutputDeclarations = (
    *RESIDUE_POINT_CLOUD_OUTPUTS[:3],
    *_GRAPH_OUTPUTS,
    *RESIDUE_POINT_CLOUD_OUTPUTS[3:],
)

_INTERFACE_RESIDUE_IDS_OUTPUT = _output(
    "interface_residue_ids",
    "interface_residue_ids",
    granularity="residue",
    layout="K",
)


def residue_graph_output_declarations(data: Mapping[str, object]) -> OutputDeclarations:
    if "interface_residue_ids" not in data:
        return RESIDUE_GRAPH_OUTPUTS
    return (*RESIDUE_GRAPH_OUTPUTS, _INTERFACE_RESIDUE_IDS_OUTPUT)


def _matrix_outputs(primary_name: str, *, binned: bool) -> OutputDeclarations:
    bin_outputs = (_output("bin_edges", "bin_edges"),) if binned else ()
    return (
        _output(primary_name, "matrix", granularity="residue", layout="NN"),
        *bin_outputs,
        *_RESIDUE_AUX_OUTPUTS,
    )


RESIDUE_DISTOGRAM_OUTPUTS = _matrix_outputs("dist", binned=False)
RESIDUE_DISTOGRAM_BINS_OUTPUTS = _matrix_outputs("dist_bins", binned=True)
RESIDUE_CONTACT_MAP_OUTPUTS = _matrix_outputs("contact", binned=False)
RESIDUE_CONTACT_MAP_BINS_OUTPUTS = _matrix_outputs("contact_bins", binned=True)

VOXEL_OUTPUTS: OutputDeclarations = (
    _output("grid", "grid", layout="CXYZ"),
    _output("grid_size", "grid_size"),
    _output("voxel_size", "voxel_size"),
    _output("origin", "origin"),
    _CENTROID_OUTPUT,
    _output("residue_ids", "residue_ids", granularity="residue", layout="N"),
    _output("chain_id", "chain_id", granularity="residue", layout="N"),
    _output("omitted_residue_ids", "omitted_residue_ids", granularity="residue", layout="N"),
    _output("resname", "residue_names", granularity="residue", layout="N"),
    _output("aa_index", "amino_acid_index", granularity="residue", layout="N"),
    _output("plddt", "confidence", granularity="residue", layout="N"),
    _output("bfactor_ca", "bfactor", granularity="residue", layout="N"),
)


__all__ = [
    "ATOM_GRAPH_OUTPUTS",
    "ATOM_POINT_CLOUD_OUTPUTS",
    "RESIDUE_CONTACT_MAP_BINS_OUTPUTS",
    "RESIDUE_CONTACT_MAP_OUTPUTS",
    "RESIDUE_DISTOGRAM_BINS_OUTPUTS",
    "RESIDUE_DISTOGRAM_OUTPUTS",
    "RESIDUE_GRAPH_OUTPUTS",
    "RESIDUE_POINT_CLOUD_OUTPUTS",
    "VOXEL_OUTPUTS",
    "residue_graph_output_declarations",
]
