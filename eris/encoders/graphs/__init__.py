"""Graph encoder registrations."""

from eris.core.config import (
    AtomKNNGraphConfig,
    ResidueContactGraphConfig,
    ResidueKNNGraphConfig,
)
from eris.core.registry import register_encoder
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph

register_encoder(
    ResidueContactGraphConfig,
    ResidueContactGraph,
    description=(
        "Residue contact graph (CA) with baseline node/edge/graph features "
        "(aa_index, physchem, plddt/bfactor, RBF, seq separation, local flags)."
    ),
)

register_encoder(
    ResidueKNNGraphConfig,
    ResidueKNNGraph,
    description=(
        "Residue kNN graph (CA) with baseline node/edge/graph features "
        "(aa_index, physchem, plddt/bfactor, RBF, seq separation, local flags)."
    ),
)

register_encoder(
    AtomKNNGraphConfig,
    AtomKNNGraph,
    description=(
        "Atom kNN graph with baseline deterministic node/edge/graph features "
        "(elem/role/bfactor/occ, dist+RBF, seq separation, local flags, graph stats)."
    ),
)
