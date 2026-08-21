from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import eris.core.base_encoder.base_atom_encoder as atom_base_module
import eris.encoders.graphs.residue_contact_graph as contact_graph_module
import eris.encoders.graphs.residue_knn_graph as residue_knn_module
from eris.core.artifacts import StructureArtifact
from eris.core.base_encoder.base_encoder import EncodingContext
from eris.core.config import (
    AtomKNNGraphConfig,
    ResidueContactGraphConfig,
    ResidueKNNGraphConfig,
)
from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph

DATA = Path("tests/data/1AQ2.pdb")
INPUT = StructureInput(source=str(DATA))
GraphEncoder = AtomKNNGraph | ResidueKNNGraph | ResidueContactGraph
GraphConfig = AtomKNNGraphConfig | ResidueKNNGraphConfig | ResidueContactGraphConfig


def _encode(encoder: GraphEncoder) -> StructureArtifact:
    return encoder.encode(INPUT)


def _assert_same_data(first: StructureArtifact, second: StructureArtifact) -> None:
    assert first.data.keys() == second.data.keys()
    for key in first.data:
        np.testing.assert_array_equal(first.data[key], second.data[key])


@pytest.fixture(scope="module")
def default_artifacts() -> dict[str, tuple[GraphEncoder, StructureArtifact]]:
    encoders: list[GraphEncoder] = [
        AtomKNNGraph(use_cache=False),
        ResidueKNNGraph(use_cache=False),
        ResidueContactGraph(use_cache=False),
    ]
    return {encoder.name: (encoder, _encode(encoder)) for encoder in encoders}


@pytest.mark.parametrize(
    ("encoder", "expected_config"),
    [
        (AtomKNNGraph(), AtomKNNGraphConfig()),
        (ResidueKNNGraph(), ResidueKNNGraphConfig()),
        (ResidueContactGraph(), ResidueContactGraphConfig()),
    ],
)
def test_default_encoder_state_is_owned_by_typed_config(
    encoder: GraphEncoder,
    expected_config: GraphConfig,
) -> None:
    assert encoder.config == expected_config
    assert set(vars(encoder)) == {"config", "use_cache", "cache"}
    assert encoder.get_params() == expected_config.to_dict()
    assert encoder.get_params()["scale"] is None


@pytest.mark.parametrize(
    ("encoder_type", "config", "kwargs"),
    [
        (
            AtomKNNGraph,
            AtomKNNGraphConfig(k=8, cutoff=None, node_attr_mode="occ+elem"),
            {"k": 8, "cutoff": None, "node_attr_mode": "occ+elem"},
        ),
        (
            ResidueKNNGraph,
            ResidueKNNGraphConfig(
                knn_mode="global",
                k=8,
                node_attr_mode="bfactor_ca+onehot",
            ),
            {
                "knn_mode": "global",
                "k": 8,
                "node_attr_mode": "bfactor_ca+onehot",
            },
        ),
        (
            ResidueContactGraph,
            ResidueContactGraphConfig(
                cutoff=10.0,
                edge_attr_mode="dist+seq",
            ),
            {"cutoff": 10.0, "edge_attr_mode": "dist+seq"},
        ),
    ],
)
def test_flat_kwargs_and_config_construction_are_equivalent(
    encoder_type: type[GraphEncoder],
    config: GraphConfig,
    kwargs: dict[str, Any],
) -> None:
    flat = encoder_type(use_cache=False, cache=object(), **kwargs)
    canonical = encoder_type(config=config, use_cache=False, cache=object())

    assert flat.config == canonical.config == config
    assert flat.get_params() == canonical.get_params()
    assert flat.fingerprint() == canonical.fingerprint()
    _assert_same_data(_encode(flat), _encode(canonical))


@pytest.mark.parametrize(
    ("name", "config", "kwargs"),
    [
        ("atom_knn_graph", AtomKNNGraphConfig(k=8), {"k": 8}),
        (
            "residue_knn_graph",
            ResidueKNNGraphConfig(knn_mode="global"),
            {"knn_mode": "global"},
        ),
        (
            "residue_contact_graph",
            ResidueContactGraphConfig(cutoff=10.0),
            {"cutoff": 10.0},
        ),
    ],
)
def test_registry_supports_config_and_flat_kwargs(
    name: str,
    config: GraphConfig,
    kwargs: dict[str, Any],
) -> None:
    flat = get_encoder(name, **kwargs)
    canonical = get_encoder(name, config=config)

    assert flat.config == canonical.config == config
    assert flat.fingerprint() == canonical.fingerprint()


@pytest.mark.parametrize(
    ("encoder_type", "config", "kwargs"),
    [
        (AtomKNNGraph, AtomKNNGraphConfig(), {"k": 8}),
        (ResidueKNNGraph, ResidueKNNGraphConfig(), {"symmetric": False}),
        (
            ResidueContactGraph,
            ResidueContactGraphConfig(),
            {"cutoff": 10.0},
        ),
    ],
)
def test_config_rejects_simultaneous_representation_kwargs(
    encoder_type: type[GraphEncoder],
    config: GraphConfig,
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(TypeError, match="config cannot be combined"):
        encoder_type(config=config, **kwargs)


def test_constructors_reject_wrong_config_types() -> None:
    with pytest.raises(TypeError, match="AtomKNNGraphConfig"):
        AtomKNNGraph(config=ResidueKNNGraphConfig())
    with pytest.raises(TypeError, match="ResidueKNNGraphConfig"):
        ResidueKNNGraph(config=ResidueContactGraphConfig())
    with pytest.raises(TypeError, match="ResidueContactGraphConfig"):
        ResidueContactGraph(config=AtomKNNGraphConfig())


@pytest.mark.parametrize(
    ("encoder_type", "kwargs", "error", "message"),
    [
        (AtomKNNGraph, {"k": 0}, ValueError, "k must be"),
        (AtomKNNGraph, {"include_self_loops": True}, TypeError, "unexpected keyword"),
        (ResidueKNNGraph, {"symmetrize": True}, TypeError, "unexpected keyword"),
        (ResidueKNNGraph, {"knn_mode": "invalid"}, ValueError, "knn_mode"),
        (ResidueContactGraph, {"symmetric": False}, TypeError, "unexpected keyword"),
        (ResidueContactGraph, {"local_cutoff": 4.5}, TypeError, "local_cutoff"),
    ],
)
def test_flat_kwargs_use_typed_config_validation(
    encoder_type: type[GraphEncoder],
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        encoder_type(**kwargs)


@pytest.mark.parametrize(
    ("encoder_type", "config"),
    [
        (AtomKNNGraph, AtomKNNGraphConfig(k=8)),
        (ResidueKNNGraph, ResidueKNNGraphConfig(knn_mode="global")),
        (ResidueContactGraph, ResidueContactGraphConfig(cutoff=10.0)),
    ],
)
def test_execution_options_are_separate_from_config_identity(
    encoder_type: type[GraphEncoder],
    config: GraphConfig,
) -> None:
    cache = object()
    operational = encoder_type(config=config, use_cache=False, cache=cache)
    default_execution = encoder_type(config=config)

    assert operational.config is config
    assert operational.use_cache is False
    assert operational.cache is cache
    assert operational.fingerprint() == default_execution.fingerprint()
    assert "use_cache" not in operational.get_params()
    assert "cache" not in operational.get_params()


def test_representation_properties_are_read_only() -> None:
    atom = AtomKNNGraph()
    residue_knn = ResidueKNNGraph()
    contact = ResidueContactGraph()

    with pytest.raises(AttributeError):
        atom.k = 8
    with pytest.raises(AttributeError):
        atom.edge_attr_mode = "dist"
    with pytest.raises(AttributeError):
        residue_knn.knn_mode = "global"
    with pytest.raises(AttributeError):
        residue_knn.include_self_loops = True
    with pytest.raises(AttributeError):
        contact.cutoff = 10.0


def test_atom_config_filters_are_passed_to_inherited_loader(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    sentinel = {"pos": np.ones((1, 3), dtype=np.float32)}

    def fake_load(inp: StructureInput, **kwargs: Any) -> dict[str, np.ndarray]:
        captured["inp"] = inp
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(atom_base_module, "load_and_extract_atoms", fake_load)
    config = AtomKNNGraphConfig(
        include_hetero=True,
        include_h=True,
        drop_water=False,
        keep_ions=False,
        altloc="max_occupancy",
        amino_acids_only=True,
    )
    encoder = AtomKNNGraph(config=config)

    assert encoder._load_structure(EncodingContext(inp=INPUT)) is sentinel
    assert captured == {
        "inp": INPUT,
        "include_hetero": True,
        "include_h": True,
        "drop_water": False,
        "keep_ions": False,
        "altloc": "max_occupancy",
        "amino_acids_only": True,
    }


@pytest.mark.parametrize(
    ("module", "encoder_type", "config_type"),
    [
        (residue_knn_module, ResidueKNNGraph, ResidueKNNGraphConfig),
        (contact_graph_module, ResidueContactGraph, ResidueContactGraphConfig),
    ],
)
@pytest.mark.parametrize("residue_center", ["ca", "mean", "com"])
def test_residue_policies_are_passed_explicitly_to_loader(
    monkeypatch,
    module: Any,
    encoder_type: type[ResidueKNNGraph] | type[ResidueContactGraph],
    config_type: type[ResidueKNNGraphConfig] | type[ResidueContactGraphConfig],
    residue_center: str,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = {"pos": np.ones((1, 3), dtype=np.float32)}

    def fake_load(inp: StructureInput, **kwargs: Any) -> dict[str, np.ndarray]:
        captured["inp"] = inp
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "load_and_extract_residue", fake_load)
    encoder = encoder_type(config=config_type(residue_center=residue_center))

    assert encoder._load_structure(EncodingContext(inp=INPUT)) is sentinel
    assert captured == {
        "inp": INPUT,
        "amino_acids_only": True,
        "mode": residue_center,
        "altloc": "first",
        "missing_ca": "skip",
    }


def test_default_features_params_and_strict_artifacts(
    default_artifacts: dict[str, tuple[GraphEncoder, StructureArtifact]],
) -> None:
    expected_widths = {
        "atom_knn_graph": 25,
        "residue_knn_graph": 38,
        "residue_contact_graph": 38,
    }

    for name, (encoder, artifact) in default_artifacts.items():
        assert artifact.data["node_attr"].shape[1] == expected_widths[name]
        assert artifact.data["edge_attr"].shape[1] == 19
        assert artifact.data["graph_attr"].shape == (13,)
        assert artifact.data["graph_attr"].dtype == np.float32
        assert artifact.meta["params"] == encoder.config.to_dict()
        assert artifact.meta["params"]["scale"] is None
        artifact.finalize()
        artifact.validate(strict=True)
        assert artifact.meta["params"] == encoder.config.to_dict()


def test_default_topology_matches_frozen_ordered_regression(
    default_artifacts: dict[str, tuple[GraphEncoder, StructureArtifact]],
) -> None:
    expected = {
        "atom_knn_graph": (
            4114,
            75900,
            "efd044e444db802d8481adacc11eb4f606aaf219db3b672362f8071e025d3fef",
        ),
        "residue_knn_graph": (
            534,
            9646,
            "9fa24bde4b6d262e6a4c85d705f7ab90d520c3bbc8d0c4e781035908c590b48c",
        ),
        "residue_contact_graph": (
            534,
            5396,
            "8b71bbae96941f43cdb375fde8930eba76147b98ba80e7c6a8955145c3886fd3",
        ),
    }

    for name, (_, artifact) in default_artifacts.items():
        n_nodes, n_edges, expected_digest = expected[name]
        edge_index = artifact.data["edge_index"]
        assert artifact.data["pos"].shape[0] == n_nodes
        assert edge_index.shape == (2, n_edges)
        assert edge_index.dtype == np.int64
        assert not np.any(edge_index[0] == edge_index[1])
        assert len({tuple(edge) for edge in edge_index.T.tolist()}) == n_edges
        assert hashlib.sha256(edge_index.tobytes()).hexdigest() == expected_digest


def test_default_edge_arrays_and_graph_statistics_remain_aligned(
    default_artifacts: dict[str, tuple[GraphEncoder, StructureArtifact]],
) -> None:
    for _, artifact in default_artifacts.values():
        edge_index = artifact.data["edge_index"]
        edge_dist = artifact.data["edge_dist"]
        edge_attr = artifact.data["edge_attr"]
        edge_keys = artifact.meta["feature_keys"]["edge_attr_keys"]
        graph_keys = artifact.meta["feature_keys"]["graph_attr_keys"]
        stats = dict(zip(graph_keys, artifact.data["graph_attr"], strict=True))

        assert edge_dist.shape == (edge_index.shape[1],)
        assert edge_attr.shape[0] == edge_index.shape[1]
        np.testing.assert_array_equal(edge_attr[:, edge_keys.index("dist")], edge_dist)
        assert stats["n_edges"] == edge_index.shape[1]
        assert stats["n_self_loops"] == 0


def test_residue_knn_modes_and_loops_preserve_topology_counts() -> None:
    local_none = _encode(ResidueKNNGraph(use_cache=False, cutoff=None))
    global_graph = _encode(ResidueKNNGraph(use_cache=False, knn_mode="global"))
    loops = _encode(ResidueKNNGraph(use_cache=False, include_self_loops=True))

    assert local_none.data["edge_index"].shape == (2, 9814)
    assert global_graph.data["edge_index"].shape == (2, 9814)
    np.testing.assert_array_equal(
        local_none.data["edge_index"],
        global_graph.data["edge_index"],
    )
    assert local_none.meta["params"]["knn_mode"] == "local"
    assert local_none.meta["params"]["cutoff"] is None
    assert global_graph.meta["params"]["knn_mode"] == "global"
    assert global_graph.meta["params"]["cutoff"] is None
    assert loops.data["edge_index"].shape == (2, 10180)
    np.testing.assert_array_equal(
        loops.data["edge_index"][:, -534:],
        np.stack([np.arange(534), np.arange(534)]),
    )


def test_contact_loops_preserve_topology_count_and_order() -> None:
    artifact = _encode(ResidueContactGraph(use_cache=False, include_self_loops=True))

    assert artifact.data["edge_index"].shape == (2, 5930)
    np.testing.assert_array_equal(
        artifact.data["edge_index"][:, -534:],
        np.stack([np.arange(534), np.arange(534)]),
    )


def test_atom_edge_mode_is_canonical_and_affects_fingerprint() -> None:
    default = AtomKNNGraph()
    distance_only = AtomKNNGraph(edge_attr_mode="dist")

    assert default.edge_attr_mode == "dist+rbf+seq+local"
    assert default.get_params()["edge_attr_mode"] == ["dist", "rbf", "seq", "local"]
    assert distance_only.get_params()["edge_attr_mode"] == ["dist"]
    assert default.fingerprint() != distance_only.fingerprint()


@pytest.mark.parametrize(
    ("encoder_type", "parameter", "value"),
    [
        (AtomKNNGraph, "k", 8),
        (AtomKNNGraph, "symmetric", False),
        (AtomKNNGraph, "rbf_k", 8),
        (ResidueKNNGraph, "include_self_loops", True),
        (ResidueKNNGraph, "cutoff", None),
        (ResidueContactGraph, "cutoff", 10.0),
        (ResidueContactGraph, "edge_attr_mode", "dist"),
    ],
)
def test_semantic_parameter_changes_fingerprint(
    encoder_type: type[GraphEncoder],
    parameter: str,
    value: Any,
) -> None:
    assert encoder_type().fingerprint() != encoder_type(**{parameter: value}).fingerprint()


def test_identity_fields_are_not_representation_parameters() -> None:
    for encoder in (AtomKNNGraph(), ResidueKNNGraph(), ResidueContactGraph()):
        assert encoder.name not in encoder.get_params()
        assert "name" not in encoder.get_params()
        assert "representation" not in encoder.get_params()

    with pytest.raises(TypeError, match="unexpected keyword"):
        ResidueKNNGraph(name="other")
    with pytest.raises(TypeError, match="unexpected keyword"):
        ResidueContactGraph(representation=object())
