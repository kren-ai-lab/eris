from morpheo.core.registry import register_encoder

from morpheo.encoders.graphs.residue_contact_graph import ResidueContactGraph
from morpheo.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from morpheo.encoders.graphs.residue_knn_graph_ns import ResidueKNNGraphNS
from morpheo.encoders.graphs.atom_knn_graph import AtomKNNGraph  # noqa: F401


register_encoder(
    "residue_contact_graph",
    lambda: ResidueContactGraph(),
    type="graph",
    level="residue",
    family="residue_graph",
    description="Residue contact graph (CA) with baseline node/edge/graph features (aa_index, physchem, plddt/bfactor, RBF, seq separation, local flags).",
    params_schema={
        "cutoff": {"type": "float", "default": 8.0},
        "include_self_loops": {"type": "bool", "default": False},
    },
)

register_encoder(
    "residue_knn_graph",
    lambda: ResidueKNNGraph(),
    type="graph",
    level="ca",
    family="residue_graph",
    description="Residue kNN graph built on CA coordinates (fixed k).",
    params_schema={
        "k": {"type": "int", "default": 16},
        "symmetrize": {"type": "bool", "default": False},
        "include_self_loops": {"type": "bool", "default": False},
    },
)

register_encoder(
    "residue_knn_graph_ns",
    lambda: ResidueKNNGraphNS(),
    type="graph",
    level="ca",
    family="residue_graph",
    description="Residue kNN graph (CA) with baseline node/edge/graph features (aa_index, physchem, plddt/bfactor, RBF, seq separation, local flags).",
    params_schema={
        "k": {"type": "int", "default": 16},
        "symmetrize": {"type": "bool", "default": False},
        "include_self_loops": {"type": "bool", "default": False},
    },
)


register_encoder(
    "atom_knn_graph_ns",
    lambda: AtomKNNGraph(),
    type="graph",
    level="atom",
    family="atom_graph",
    description="Atom kNN graph with baseline deterministic node/edge/graph features (elem/role/bfactor/occ, dist+RBF, seq separation, local flags, graph stats).",
    params_schema={
        "k": {"type": "int", "default": 16},
        "include_self_loops": {"type": "bool", "default": False},
        "include_hetero": {"type": "bool", "default": False},
        "include_h": {"type": "bool", "default": False},
        "altloc": {"type": "str", "default": "first"},
        "amino_acids_only": {"type": "bool", "default": False},
        "rbf_k": {"type": "int", "default": 16},
        "rbf_dmax": {"type": "float", "default": 20.0},
        "local_cutoff": {"type": "int", "default": 4},
    },
)
