from __future__ import annotations

import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest


def _torch():
    return pytest.importorskip("torch")


def _metadata(
    *,
    name: str,
    family: str,
    representation_type: str,
    sample_id: str = "sample",
) -> dict[str, object]:
    return {
        "id": sample_id,
        "artifact_dir": f"artifacts/{sample_id}/{name}",
        "meta_path": f"artifacts/{sample_id}/{name}/meta.json",
        "representation_name": name,
        "representation_family": family,
        "representation_type": representation_type,
        "granularity": "residue",
        "parameters": {"representation": name, "k": 8},
        "feature_keys": {
            "node_attr_keys": ["aa_index"],
            "edge_attr_keys": ["edge.distance"],
            "graph_attr_keys": ["graph.n_nodes"],
        },
        "outputs": [{"name": "x"}, {"name": "edge_index"}],
    }


def _graph_sample(sample_id: str, *, node_offset: float = 0.0) -> dict[str, object]:
    torch = _torch()
    return {
        **_metadata(
            name="residue_knn_graph",
            family="explicit",
            representation_type="graph",
            sample_id=sample_id,
        ),
        "x": torch.tensor([[1.0 + node_offset], [2.0 + node_offset]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        "edge_attr": torch.tensor([[0.5], [0.6]]),
        "edge_dist": torch.tensor([3.0, 4.0]),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "graph_attr": torch.tensor([2.0]),
    }


def _cloud_sample(sample_id: str, n_points: int) -> dict[str, object]:
    torch = _torch()
    return {
        **_metadata(
            name="residue_point_cloud",
            family="explicit",
            representation_type="cloud",
            sample_id=sample_id,
        ),
        "pos": torch.arange(n_points * 3, dtype=torch.float32).reshape(n_points, 3),
        "x": torch.ones((n_points, 2), dtype=torch.float32),
        "centroid": torch.tensor([1.0, 2.0, 3.0]),
    }


def _matrix_sample(sample_id: str, n: int = 2) -> dict[str, object]:
    torch = _torch()
    return {
        **_metadata(
            name="residue_contact_map",
            family="explicit",
            representation_type="matrix",
            sample_id=sample_id,
        ),
        "x": torch.eye(n, dtype=torch.bool).unsqueeze(0),
        "matrix_key": "contact",
        "n": n,
        "bin_edges": torch.tensor([0.0, 1.0], dtype=torch.float32),
    }


def _grid_sample(sample_id: str, size: int = 2) -> dict[str, object]:
    torch = _torch()
    return {
        **_metadata(
            name="residue_voxel_grid",
            family="explicit",
            representation_type="grid",
            sample_id=sample_id,
        ),
        "x": torch.zeros((2, size, size, size), dtype=torch.float32),
        "grid_size": torch.tensor([size], dtype=torch.int32),
        "origin": torch.tensor([float(size), 0.0, 0.0]),
        "centroid": torch.tensor([0.0, float(size), 0.0]),
    }


def _learned_sample(
    sample_id: str,
    *,
    node: bool,
    global_: bool,
    n_nodes: int = 2,
    dim: int = 3,
    dtype=np.float32,
    valid_mask: bool = True,
) -> dict[str, object]:
    sample = _metadata(
        name="gearnet_edge",
        family="learned",
        representation_type="embedding",
        sample_id=sample_id,
    )
    outputs = []
    if node:
        sample["node_embedding"] = np.arange(n_nodes * dim, dtype=dtype).reshape(n_nodes, dim)
        sample["node_ids"] = np.asarray([f"A:{index + 1}::GLY" for index in range(n_nodes)])
        outputs.extend([{"name": "node_embedding"}, {"name": "node_ids"}])
        if valid_mask:
            sample["valid_mask"] = np.asarray([True] * n_nodes, dtype=np.bool_)
            outputs.append({"name": "valid_mask"})
    if global_:
        sample["global_embedding"] = np.arange(dim, dtype=dtype)
        outputs.append({"name": "global_embedding"})
    sample["outputs"] = outputs
    sample["feature_keys"] = {}
    return sample


def test_collate_graph_offsets_edges_and_preserves_metadata() -> None:
    torch = _torch()
    from eris.datasets import collate_graph

    out = collate_graph([_graph_sample("a"), _graph_sample("b", node_offset=10.0)])

    assert out["id"] == ["a", "b"]
    assert out["representation_name"] == "residue_knn_graph"
    assert out["x"].shape == (4, 1)
    assert torch.equal(out["batch"], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(out["ptr"], torch.tensor([0, 2, 4]))
    assert torch.equal(
        out["edge_index"],
        torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int64),
    )
    assert out["graph_attr"].shape == (2, 1)
    assert out["parameters"] == {"representation": "residue_knn_graph", "k": 8}


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda sample: sample.pop("edge_attr"), "partial optional-field"),
        (lambda sample: sample["x"].resize_(2, 2), "feature width mismatch"),
        (lambda sample: sample["edge_attr"].resize_(2, 2), "feature width mismatch"),
        (lambda sample: sample["graph_attr"].resize_(2), "shape mismatch"),
        (lambda sample: sample.update({"parameters": {"representation": "other"}}), "parameters"),
        (
            lambda sample: sample.update({"feature_keys": {"node_attr_keys": ["other"]}}),
            "feature_keys",
        ),
        (
            lambda sample: sample.update({"representation_name": "residue_contact_graph"}),
            "mixed representation",
        ),
    ],
)
def test_collate_graph_rejects_incompatible_samples(mutator, match: str) -> None:
    from eris.datasets import collate_graph

    first = _graph_sample("a")
    second = _graph_sample("b")
    mutator(second)

    with pytest.raises(ValueError, match=match):
        collate_graph([first, second])


def test_collate_graph_rejects_empty_batch_none_and_dtype_mismatch() -> None:
    torch = _torch()
    from eris.datasets import collate_graph

    with pytest.raises(ValueError, match="non-empty"):
        collate_graph([])
    with pytest.raises(ValueError, match="must not be None"):
        collate_graph([None])  # type: ignore[list-item]

    first = _graph_sample("a")
    second = _graph_sample("b")
    second["x"] = torch.ones((2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        collate_graph([first, second])


def test_collate_graph_strict_is_thin_alias() -> None:
    from eris.datasets.collate import collate_graph_strict

    out = collate_graph_strict([_graph_sample("a"), _graph_sample("b")])

    assert out["ptr"].tolist() == [0, 2, 4]


def test_collate_cloud_concatenates_variable_n_without_padding() -> None:
    torch = _torch()
    from eris.datasets import collate_cloud

    out = collate_cloud([_cloud_sample("a", 1), _cloud_sample("b", 3)])

    assert out["pos"].shape == (4, 3)
    assert out["x"].shape == (4, 2)
    assert torch.equal(out["batch"], torch.tensor([0, 1, 1, 1]))
    assert torch.equal(out["ptr"], torch.tensor([0, 1, 4]))
    assert out["centroid"].shape == (2, 3)


def test_collate_cloud_rejects_partial_x_and_width_mismatch() -> None:
    from eris.datasets import collate_cloud

    with pytest.raises(ValueError, match="partial optional-field"):
        sample = _cloud_sample("b", 2)
        sample.pop("x")
        collate_cloud([_cloud_sample("a", 1), sample])

    sample = _cloud_sample("b", 2)
    sample["x"] = _torch().ones((2, 3), dtype=_torch().float32)
    with pytest.raises(ValueError, match="feature width mismatch"):
        collate_cloud([_cloud_sample("a", 1), sample])


def test_collate_matrix_stacks_identical_shapes_without_padding() -> None:
    from eris.datasets import collate_matrix

    out = collate_matrix([_matrix_sample("a"), _matrix_sample("b")])

    assert out["x"].shape == (2, 1, 2, 2)
    assert out["matrix_key"] == "contact"
    assert out["n"] == [2, 2]


def test_collate_matrix_rejects_shape_matrix_key_and_bin_edge_mismatch() -> None:
    torch = _torch()
    from eris.datasets import collate_matrix

    with pytest.raises(ValueError, match="shape mismatch"):
        collate_matrix([_matrix_sample("a", 2), _matrix_sample("b", 3)])

    sample = _matrix_sample("b")
    sample["matrix_key"] = "dist"
    with pytest.raises(ValueError, match="matrix_key mismatch"):
        collate_matrix([_matrix_sample("a"), sample])

    sample = _matrix_sample("b")
    sample["bin_edges"] = torch.tensor([0.0, 2.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="bin_edges mismatch"):
        collate_matrix([_matrix_sample("a"), sample])


def test_collate_grid_stacks_identical_shapes_and_spatial_metadata() -> None:
    from eris.datasets import collate_grid

    out = collate_grid([_grid_sample("a"), _grid_sample("b")])

    assert out["x"].shape == (2, 2, 2, 2, 2)
    assert out["grid_size"].shape == (2, 1)
    assert out["origin"].shape == (2, 3)
    assert out["centroid"].shape == (2, 3)


def test_collate_grid_rejects_shape_and_grid_size_mismatch() -> None:
    from eris.datasets import collate_grid

    with pytest.raises(ValueError, match="shape mismatch"):
        collate_grid([_grid_sample("a", 2), _grid_sample("b", 3)])

    sample = _grid_sample("b")
    sample["grid_size"] = _torch().tensor([9], dtype=_torch().int32)
    with pytest.raises(ValueError, match="grid_size"):
        collate_grid([_grid_sample("a"), sample])


def test_collate_learned_node_preserves_node_ids_mask_and_order() -> None:
    torch = _torch()
    from eris.datasets import collate_learned

    out = collate_learned(
        [
            _learned_sample("a", node=True, global_=False, n_nodes=1),
            _learned_sample("b", node=True, global_=False, n_nodes=2),
        ]
    )

    assert out["node_embedding"].shape == (3, 3)
    assert out["node_embedding"].dtype == torch.float32
    assert out["node_ids"] == ["A:1::GLY", "A:1::GLY", "A:2::GLY"]
    assert torch.equal(out["valid_mask"], torch.tensor([True, True, True]))
    assert torch.equal(out["batch"], torch.tensor([0, 1, 1]))
    assert torch.equal(out["ptr"], torch.tensor([0, 1, 3]))


def test_collate_learned_global_stacks_embeddings() -> None:
    from eris.datasets import collate_learned

    out = collate_learned(
        [
            _learned_sample("a", node=False, global_=True),
            _learned_sample("b", node=False, global_=True),
        ]
    )

    assert out["global_embedding"].shape == (2, 3)
    assert out["id"] == ["a", "b"]


def test_collate_learned_both_preserves_node_global_association() -> None:
    from eris.datasets import collate_learned

    out = collate_learned(
        [
            _learned_sample("a", node=True, global_=True, n_nodes=1),
            _learned_sample("b", node=True, global_=True, n_nodes=2),
        ]
    )

    assert out["global_embedding"].shape == (2, 3)
    assert out["node_embedding"].shape == (3, 3)
    assert out["id"] == ["a", "b"]
    assert out["ptr"].tolist() == [0, 1, 3]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda sample: sample.update({"global_embedding": np.ones((4,), dtype=np.float32)}),
            "shape mismatch",
        ),
        (
            lambda sample: sample.update({"node_embedding": np.ones((2, 4), dtype=np.float32)}),
            "feature width mismatch",
        ),
        (
            lambda sample: sample.update({"node_embedding": np.ones((2, 3), dtype=np.float64)}),
            "dtype mismatch",
        ),
        (lambda sample: sample.pop("valid_mask"), "partial optional-field"),
        (lambda sample: sample.update({"node_ids": ["only-one"]}), "node_ids"),
    ],
)
def test_collate_learned_rejects_incompatible_node_outputs(mutator, match: str) -> None:
    from eris.datasets import collate_learned

    first = _learned_sample("a", node=True, global_=True)
    second = _learned_sample("b", node=True, global_=True)
    mutator(second)

    with pytest.raises(ValueError, match=match):
        collate_learned([first, second])


def test_collate_rejects_explicit_learned_mixed_batch() -> None:
    from eris.datasets import collate_graph

    learned = _learned_sample("b", node=True, global_=False)

    with pytest.raises(ValueError, match="mixed representation"):
        collate_graph([_graph_sample("a"), learned])


def test_padding_helpers_remain_explicit_opt_in() -> None:
    from eris.datasets import collate_cloud_pad, collate_grid_pad

    cloud = collate_cloud_pad([_cloud_sample("a", 1), _cloud_sample("b", 3)])
    grid = collate_grid_pad([_grid_sample("a", 2), _grid_sample("b", 3)], return_mask=True)

    assert cloud["mask"].shape == (2, 3)
    assert grid["mask"].shape == (2, 1, 3, 3, 3)


def test_import_eris_datasets_does_not_eagerly_import_torch() -> None:
    script = """
import sys
import eris.datasets

blocked = {"torch", "torch_geometric", "torchdrug", "torch_scatter", "torch_cluster"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_common_metadata_is_compared_after_normalization() -> None:
    from eris.datasets import collate_graph

    first = _graph_sample("a")
    second = _graph_sample("b")
    first["parameters"] = {"nested": {"b": 2, "a": [1]}}
    second["parameters"] = {"nested": {"a": [1], "b": 2}}

    out = collate_graph([first, second])

    assert out["parameters"] == {"nested": {"b": 2, "a": [1]}}


def test_samples_are_not_mutated_by_collate() -> None:
    from eris.datasets import collate_graph

    sample = _graph_sample("a")
    before = deepcopy(sample)

    collate_graph([sample, _graph_sample("b")])

    assert sample.keys() == before.keys()
