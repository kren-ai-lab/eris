from __future__ import annotations

from pathlib import Path

import numpy as np

from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import StructureArtifact


def save_graph_artifact_v2(
    target: Path,
    *,
    graph_attr: tuple[float, float],
) -> StructureArtifact:
    """Build, strictly validate, and save one minimal explicit graph artifact."""

    data = {
        "node_attr": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "graph_attr": np.asarray(graph_attr, dtype=np.float32),
        "chain_id": np.asarray(["A", "A"], dtype=np.str_),
    }
    parameters = {"fixture": "minimal_graph_attr"}
    metadata = build_artifact_metadata(
        data=data,
        representation_name="test_graph_fixture",
        representation_family="explicit",
        representation_type="graph",
        granularity="residue",
        coordinate_source="ca",
        encoder={
            "implementation": "tests.fixtures.MinimalGraphProducer",
            "version": "1",
        },
        input_metadata=build_input_metadata(
            source="memory://minimal-graph",
            source_sha256="a" * 64,
            input_format="pdb",
            model_index=0,
            chain_ids=["A"],
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="test_graph_fixture",
            parameters=parameters,
            chain_ids=["A"],
        ),
        output_declarations=[
            {
                "name": "node_attr",
                "role": "node_features",
                "granularity": "residue",
                "layout": "NF",
            },
            {"name": "edge_index", "role": "edge_index", "layout": "2E"},
            {
                "name": "graph_attr",
                "role": "graph_features",
                "granularity": "structure",
                "layout": "F",
            },
            {"name": "chain_id", "role": "chain_id", "granularity": "residue", "layout": "N"},
        ],
        feature_keys=build_feature_keys(
            node_attr_keys=["node.first", "node.second"],
            graph_attr_keys=["graph.n_nodes", "graph.n_edges"],
        ),
        counts={"n_nodes": 2, "n_edges": 2, "n_self_loops": 0},
    )
    artifact = StructureArtifact(data=data, meta=metadata)
    artifact.validate(strict=True)
    artifact.save(target)
    return artifact


__all__ = ["save_graph_artifact_v2"]
