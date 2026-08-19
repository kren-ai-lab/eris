from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.registry import get_encoder
from eris.core.types import StructureInput


DATA = Path("tests/data/1AQ2.pdb")
MISSING = object()


def _artifact(
    data: dict[str, np.ndarray],
    *,
    feature_keys: Any = MISSING,
    legacy: dict[str, Any] | None = None,
) -> StructureArtifact:
    meta: dict[str, Any] = {
        "encoder_name": "test_encoder",
        "encoder_version": "0.1.0",
        "encoder_fingerprint": "test",
        "type": "graph",
        "level": "residue",
        "params": {},
        "input": {
            "source": "memory",
            "chain_id": None,
            "model_index": 0,
            "format": None,
        },
    }
    if feature_keys is not MISSING:
        meta["feature_keys"] = feature_keys
    if legacy:
        meta.update(legacy)
    return StructureArtifact(data=data, meta=meta)


def _complete_graph_data() -> dict[str, np.ndarray]:
    return {
        "node_attr": np.zeros((2, 2), dtype=np.float32),
        "edge_attr": np.zeros((3, 2), dtype=np.float32),
        "graph_attr": np.zeros((2,), dtype=np.float32),
    }


def _complete_feature_keys() -> dict[str, list[str]]:
    return {
        "node_attr_keys": ["node.z", "node.a"],
        "edge_attr_keys": ["edge.z", "edge.a"],
        "graph_attr_keys": ["graph.z", "graph.a"],
    }


def test_explicit_feature_keys_survive_finalize_in_exact_order():
    expected = _complete_feature_keys()
    artifact = _artifact(_complete_graph_data(), feature_keys=deepcopy(expected))

    artifact.finalize()

    assert artifact.meta["feature_keys"] == expected
    assert artifact.meta["feature_keys"]["node_attr_keys"] == ["node.z", "node.a"]
    assert artifact.meta["feature_keys"]["edge_attr_keys"] == ["edge.z", "edge.a"]
    assert artifact.meta["feature_keys"]["graph_attr_keys"] == ["graph.z", "graph.a"]


def test_finalize_preserves_authoritative_feature_keys_idempotently():
    artifact = _artifact(
        _complete_graph_data(),
        feature_keys=_complete_feature_keys(),
    )

    artifact.finalize()
    after_first = deepcopy(artifact.meta["feature_keys"])
    artifact.finalize()

    assert artifact.meta["feature_keys"] == after_first


def test_partial_explicit_feature_keys_win_over_conflicting_legacy_fallback():
    explicit = {"node_attr_keys": ["explicit"]}
    artifact = _artifact(
        {"node_attr": np.zeros((1, 1), dtype=np.float32)},
        feature_keys=deepcopy(explicit),
        legacy={
            "node_attr_keys": ["legacy"],
            "edge_attr_keys": ["legacy.edge"],
        },
    )

    artifact.finalize()

    assert artifact.meta["feature_keys"] == explicit


def test_missing_feature_keys_use_existing_legacy_fallback():
    artifact = _artifact(
        {
            "node_attr": np.zeros((1, 1), dtype=np.float32),
            "graph_attr": np.zeros((2,), dtype=np.float32),
            "graph_attr_keys": np.asarray(["graph.0", "graph.1"]),
        },
        legacy={"node_attr_keys": ["node.0"]},
    )

    artifact.finalize()

    assert artifact.meta["feature_keys"] == {
        "node_attr_keys": ["node.0"],
        "graph_attr_keys": ["graph.0", "graph.1"],
    }


def test_empty_feature_keys_use_existing_legacy_fallback():
    artifact = _artifact(
        {"edge_attr": np.zeros((2, 1), dtype=np.float32)},
        feature_keys={},
        legacy={"edge_attr_keys": ["edge.0"]},
    )

    artifact.finalize()

    assert artifact.meta["feature_keys"] == {"edge_attr_keys": ["edge.0"]}


@pytest.mark.parametrize("feature_keys", [["bad"], "bad"])
def test_non_dict_explicit_feature_keys_fail(feature_keys):
    artifact = _artifact({}, feature_keys=feature_keys)

    with pytest.raises(ArtifactValidationError, match="meta.feature_keys must be a dict"):
        artifact.finalize()


@pytest.mark.parametrize(
    "node_attr_keys",
    ["not-a-list", ["valid", 1]],
)
def test_invalid_feature_key_field_types_fail(node_attr_keys):
    artifact = _artifact(
        {"node_attr": np.zeros((1, 2), dtype=np.float32)},
        feature_keys={"node_attr_keys": node_attr_keys},
    )

    with pytest.raises(
        ArtifactValidationError,
        match=r"meta\.feature_keys\.node_attr_keys must be a list\[str\]",
    ):
        artifact.finalize()


@pytest.mark.parametrize(
    ("data", "feature_keys", "message"),
    [
        (
            {"node_attr": np.zeros((2, 2), dtype=np.float32)},
            {"node_attr_keys": ["node.0"]},
            r"node_attr_keys has 1 keys.*2 columns",
        ),
        (
            {"edge_attr": np.zeros((3, 2), dtype=np.float32)},
            {"edge_attr_keys": ["edge.0"]},
            r"edge_attr_keys has 1 keys.*2 columns",
        ),
        (
            {"graph_attr": np.zeros((2,), dtype=np.float32)},
            {"graph_attr_keys": ["graph.0"]},
            r"graph_attr_keys has 1 keys.*length 2",
        ),
    ],
)
def test_feature_key_cardinality_mismatches_fail(data, feature_keys, message):
    artifact = _artifact(data, feature_keys=feature_keys)

    with pytest.raises(ArtifactValidationError, match=message):
        artifact.finalize()


def test_real_graph_encoder_keys_survive_finalize():
    encoder = get_encoder("residue_contact_graph")
    encoder.use_cache = False
    artifact = encoder.encode(StructureInput(source=str(DATA)))
    expected = deepcopy(artifact.meta["feature_keys"])

    artifact.finalize()

    assert artifact.meta["feature_keys"] == expected
    assert list(expected) == [
        "node_attr_keys",
        "edge_attr_keys",
        "graph_attr_keys",
    ]
    assert len(expected["node_attr_keys"]) == artifact.data["node_attr"].shape[1]
    assert len(expected["edge_attr_keys"]) == artifact.data["edge_attr"].shape[1]
    assert len(expected["graph_attr_keys"]) == artifact.data["graph_attr"].shape[0]


def test_authoritative_feature_keys_survive_save_load_roundtrip(tmp_path):
    expected = _complete_feature_keys()
    artifact = _artifact(
        _complete_graph_data(),
        feature_keys=deepcopy(expected),
    )
    artifact_dir = tmp_path / "artifact"

    artifact.save(artifact_dir)
    loaded = StructureArtifact.load(artifact_dir, validate=True, strict=True)

    assert loaded.meta["feature_keys"] == expected
