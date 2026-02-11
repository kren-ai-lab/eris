from morpheo.core.registry import register_encoder

from morpheo.encoders.graphs.residue_contact_graph import ResidueContactGraph
from morpheo.encoders.graphs.residue_contact_graph_ns import ResidueContactGraphNS
from morpheo.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from morpheo.encoders.graphs.residue_knn_graph_ns import ResidueKNNGraphNS
from morpheo.encoders.graphs.atom_knn_graph import AtomKNNGraph
from morpheo.encoders.graphs.atom_knn_graph_ns import AtomKNNGraphNS  # noqa: F401

register_encoder(
    "residue_contact_graph",
    lambda: ResidueContactGraph(),
    type="graph",
    level="ca",
    family="residue",
    desc="Residue contact graph from CA distances (thresholded).",
)

register_encoder(
    "residue_contact_graph_ns",
    lambda: ResidueContactGraphNS(),
    type="graph",
    level="ca",
    family="residue",
    desc="Residue contact graph (CA) with node/edge features baseline (e.g., aa_index, plddt/bfactor, RBF edge).",
)

register_encoder(
    "residue_knn_graph",
    lambda: ResidueKNNGraph(),
    type="graph",
    level="ca",
    family="residue",
    desc="Residue kNN graph built on CA coordinates (fixed k).",
)

register_encoder(
    "residue_knn_graph_ns",
    lambda: ResidueKNNGraphNS(),
    type="graph",
    level="ca",
    family="residue",
    desc="Residue kNN graph (CA) with node/edge features baseline (e.g., aa_index, plddt/bfactor, RBF edge).",
)

### Checking
register_encoder(
    "atom_knn_graph",
    lambda: AtomKNNGraph(),
    type="graph",
    level="atom",
    family="atom",
    desc="Atom KNN graph.",
)

register_encoder(
    "atom_knn_graph_ns",
    lambda: AtomKNNGraphNS(),
    type="graph",
    level="atom",
    family="atom",
    desc="Atom KNN graph NS.",
)
