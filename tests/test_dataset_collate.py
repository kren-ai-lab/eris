from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from eris.core.artifact_schema import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.datasets.collate import collate_cloud, collate_graph, collate_grid, collate_matrix
from eris.datasets.torch_dataset import DatasetIndexError, ErisFolderDataset
from eris.pipelines.batch_encode import batch_encode
from eris.pipelines.manifest import StructureManifestRow
from eris.pipelines.pack import PACKED_INDEX_FIELDS, pack_artifacts

PDB = Path("tests/data/1AQ2.pdb")


def _write_index(
    path: Path,
    rows: list[dict[str, object]],
    *,
    extra: list[str] | None = None,
) -> None:
    fieldnames = [*PACKED_INDEX_FIELDS, *(extra or [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_row(path: Path, **overrides: object) -> dict[str, object]:
    row = _read_index(path)[0]
    row.update(overrides)
    return row


def _artifact_metadata(
    *,
    data: dict[str, np.ndarray],
    representation_name: str,
    representation_family: str = "explicit",
    representation_type: str,
    granularity: str = "residue",
    counts: dict[str, int] | None = None,
    outputs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if outputs is None:
        outputs = [
            {
                "name": next(iter(data)),
                "role": "coordinates",
                "granularity": granularity,
                "layout": "NC",
            }
        ]
    parameters = {"fixture": representation_name}
    requires_chain_id = granularity == "residue" or (
        granularity == "mixed" and "node_embedding" in data
    )
    if requires_chain_id and "chain_id" not in data:
        node_count = (
            int(data["node_embedding"].shape[0])
            if "node_embedding" in data
            else int(data[next(iter(data))].shape[0])
        )
        if representation_type == "grid":
            node_count = 1
        data["chain_id"] = np.full((node_count,), "A", dtype=np.str_)
        outputs.append(
            {
                "name": "chain_id",
                "role": "chain_id",
                "granularity": "residue",
                "layout": "N",
            }
        )
    kwargs: dict[str, object] = {}
    if representation_family == "learned":
        kwargs["model"] = {
            "architecture": "SyntheticModel",
            "checkpoint": {
                "identifier": "synthetic.pth",
                "revision": "test-revision",
                "source": "memory://checkpoint",
                "sha256": "f" * 64,
            },
        }

    return build_artifact_metadata(
        data=data,
        representation_name=representation_name,
        representation_family=representation_family,
        representation_type=representation_type,
        granularity=granularity,
        coordinate_source="model_specific" if representation_family == "learned" else "ca",
        encoder={
            "implementation": "tests.fixtures.DatasetProducer",
            "version": "1",
        },
        input_metadata=build_input_metadata(
            source="memory://dataset",
            source_sha256="a" * 64,
            input_format="pdb",
            model_index=0,
            chain_ids=["A"],
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name=representation_name,
            parameters=parameters,
            chain_ids=["A"],
        ),
        output_declarations=outputs,
        feature_keys=build_feature_keys(),
        counts={} if counts is None else counts,
        **kwargs,
    )


def _save_artifact(
    target: Path,
    *,
    representation_name: str = "residue_point_cloud",
    representation_family: str = "explicit",
    representation_type: str = "cloud",
    granularity: str = "residue",
    data: dict[str, np.ndarray] | None = None,
    outputs: list[dict[str, object]] | None = None,
    counts: dict[str, int] | None = None,
) -> StructureArtifact:
    if data is None:
        data = {"pos": np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)}
    artifact = StructureArtifact(
        data=data,
        meta=_artifact_metadata(
            data=data,
            representation_name=representation_name,
            representation_family=representation_family,
            representation_type=representation_type,
            granularity=granularity,
            outputs=outputs,
            counts=counts,
        ),
    )
    artifact.save(target)
    return artifact


def _save_graph_artifact(target: Path) -> StructureArtifact:
    return _save_artifact(
        target,
        representation_name="residue_contact_graph",
        representation_type="graph",
        data={
            "node_attr": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            "graph_attr": np.asarray([2.0], dtype=np.float32),
        },
        outputs=[
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
        ],
        counts={"n_nodes": 2, "n_edges": 2},
    )


def _save_matrix_artifact(target: Path) -> StructureArtifact:
    return _save_artifact(
        target,
        representation_name="residue_contact_map",
        representation_type="matrix",
        data={"contact": np.asarray([[True]], dtype=np.bool_)},
        outputs=[
            {
                "name": "contact",
                "role": "contact_matrix",
                "granularity": "residue",
                "layout": "NN",
            }
        ],
        counts={"n_nodes": 1},
    )


def _save_grid_artifact(target: Path) -> StructureArtifact:
    return _save_artifact(
        target,
        representation_name="residue_voxel_grid",
        representation_type="grid",
        data={"grid": np.zeros((1, 2, 2, 2), dtype=np.float32)},
        outputs=[
            {
                "name": "grid",
                "role": "grid",
                "granularity": "residue",
                "layout": "CXYZ",
            }
        ],
    )


def _save_learned_artifact(target: Path) -> StructureArtifact:
    return _save_artifact(
        target,
        representation_name="synthetic_learned_embedding",
        representation_family="learned",
        representation_type="embedding",
        granularity="mixed",
        data={
            "node_embedding": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            "global_embedding": np.asarray([0.5, 0.6], dtype=np.float32),
            "node_ids": np.asarray(["A:1::ALA", "A:2::GLY"], dtype=np.str_),
        },
        outputs=[
            {"name": "node_embedding", "role": "node_embedding", "layout": "ND"},
            {"name": "global_embedding", "role": "global_embedding", "layout": "D"},
            {
                "name": "node_ids",
                "role": "node_ids",
                "granularity": "residue",
                "layout": "N",
            },
        ],
        counts={"n_nodes": 2},
    )


def _pack(root: Path, index_path: Path) -> list[dict[str, str]]:
    pack_artifacts(root, index_path)
    return _read_index(index_path)


def test_dataset_accepts_canonical_pack_index_and_preserves_len_order(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_artifact(root / "b" / "residue_point_cloud")
    _save_artifact(root / "a" / "residue_point_cloud")
    index = tmp_path / "index.csv"
    _pack(root, index)

    dataset = ErisFolderDataset(index)

    assert len(dataset) == 2
    assert [row.id for row in dataset.rows] == ["a", "b"]


def test_dataset_rejects_missing_required_columns(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    index.write_text("id,artifact_dir\nsample,path\n", encoding="utf-8")

    with pytest.raises(DatasetIndexError, match="missing required columns"):
        ErisFolderDataset(index)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("id", "", "non-empty"),
        ("representation_family", "other", "representation_family"),
        ("shape_summary", "not json", "shape_summary"),
    ],
)
def test_dataset_rejects_malformed_rows(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    root = tmp_path / "artifacts"
    _save_artifact(root / "sample" / "residue_point_cloud")
    index = tmp_path / "index.csv"
    _pack(root, index)
    row = _index_row(index, **{field: value})
    _write_index(index, [row])

    with pytest.raises(DatasetIndexError, match=match):
        ErisFolderDataset(index)


def test_dataset_allows_extra_columns_after_canonical_header_but_ignores_them(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _save_learned_artifact(root / "sample" / "synthetic_learned_embedding")
    index = tmp_path / "index.csv"
    row = _pack(root, index)[0]
    row["encoder"] = "legacy"
    row["n_residues"] = "legacy"
    row["grid_shape"] = "legacy"
    _write_index(index, [row], extra=["encoder", "n_residues", "grid_shape"])

    dataset = ErisFolderDataset(index)
    item = dataset[0]

    assert item["representation_name"] == "synthetic_learned_embedding"
    assert "encoder" not in item
    assert "n_residues" not in item
    assert "grid_shape" not in item


def test_dataset_rejects_duplicate_identity_and_artifact_dir(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    first = root / "first" / "residue_point_cloud"
    second = root / "second" / "residue_point_cloud"
    _save_artifact(first)
    _save_artifact(second)
    index = tmp_path / "index.csv"
    rows = _pack(root, index)

    duplicate_identity = [dict(rows[0]), dict(rows[1], id=rows[0]["id"])]
    _write_index(index, duplicate_identity)
    with pytest.raises(DatasetIndexError, match="duplicate artifact"):
        ErisFolderDataset(index)

    duplicate_dir = [dict(rows[0]), dict(rows[1], artifact_dir=rows[0]["artifact_dir"])]
    duplicate_dir[1]["meta_path"] = rows[0]["meta_path"]
    _write_index(index, duplicate_dir)
    with pytest.raises(DatasetIndexError, match="duplicate artifact_dir"):
        ErisFolderDataset(index)


@pytest.mark.parametrize(
    "index_factory,expected",
    [
        (lambda tmp, root: tmp / "index.csv", "artifacts/sample/residue_point_cloud"),
        (
            lambda tmp, root: tmp / "collections" / "index.csv",
            "../artifacts/sample/residue_point_cloud",
        ),
    ],
)
def test_dataset_resolves_posix_locators_from_index_parent(
    tmp_path: Path,
    index_factory,
    expected: str,
) -> None:
    root = tmp_path / "artifacts"
    artifact_dir = root / "sample" / "residue_point_cloud"
    _save_learned_artifact(artifact_dir)
    index = index_factory(tmp_path, root)
    rows = _pack(root, index)

    dataset = ErisFolderDataset(index)
    item = dataset[0]

    assert rows[0]["artifact_dir"] == expected
    assert Path(item["artifact_dir"]).resolve() == artifact_dir.resolve()


def test_dataset_does_not_rescan_after_index_creation(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_learned_artifact(root / "indexed" / "synthetic_learned_embedding")
    index = tmp_path / "index.csv"
    _pack(root, index)
    _save_learned_artifact(root / "not-indexed" / "synthetic_learned_embedding")

    dataset = ErisFolderDataset(index)

    assert len(dataset) == 1
    assert dataset[0]["id"] == "indexed"


def test_dataset_rejects_missing_and_corrupt_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    artifact_dir = root / "sample" / "residue_point_cloud"
    _save_artifact(artifact_dir)
    index = tmp_path / "index.csv"
    _pack(root, index)
    (artifact_dir / "data.npz").unlink()
    dataset = ErisFolderDataset(index)

    with pytest.raises(ArtifactValidationError, match="Missing artifact files"):
        dataset[0]

    _save_artifact(artifact_dir)
    (artifact_dir / "meta.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="Invalid artifact metadata JSON"):
        dataset[0]


def test_dataset_rejects_meta_path_inconsistency(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_artifact(root / "sample" / "residue_point_cloud")
    index = tmp_path / "index.csv"
    row = _pack(root, index)[0]
    row["meta_path"] = "artifacts/other/meta.json"
    _write_index(index, [row])

    with pytest.raises(DatasetIndexError, match="inconsistent meta_path"):
        ErisFolderDataset(index)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("representation_name", "atom_point_cloud", "representation_name"),
        ("n_nodes", "999", "n_nodes"),
        ("shape_summary", '{"other":[1]}', "shape_summary"),
    ],
)
def test_dataset_rejects_stale_index_metadata(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    root = tmp_path / "artifacts"
    _save_artifact(root / "sample" / "residue_point_cloud", counts={"n_nodes": 1})
    index = tmp_path / "index.csv"
    row = _pack(root, index)[0]
    row[field] = value
    _write_index(index, [row])
    dataset = ErisFolderDataset(index)

    with pytest.raises(DatasetIndexError, match=match):
        dataset[0]


def test_dataset_rejects_stale_n_edges_when_index_declares_it(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_graph_artifact(root / "sample" / "residue_contact_graph")
    index = tmp_path / "index.csv"
    row = _pack(root, index)[0]
    row["n_edges"] = "999"
    _write_index(index, [row])
    dataset = ErisFolderDataset(index)

    with pytest.raises(DatasetIndexError, match="n_edges"):
        dataset[0]


@pytest.mark.parametrize(
    ("representation", "collate", "expected_shape"),
    [
        ("graph", collate_graph, (4, 1)),
        ("cloud", collate_cloud, (2, 3)),
        ("matrix", collate_matrix, (2, 1, 1, 1)),
        ("grid", collate_grid, (2, 1, 2, 2, 2)),
    ],
)
def test_dataset_samples_feed_each_canonical_explicit_collate(
    tmp_path: Path,
    representation: str,
    collate,
    expected_shape: tuple[int, ...],
) -> None:
    root = tmp_path / "artifacts"
    for sample_id in ("first", "second"):
        target = root / sample_id
        if representation == "graph":
            _save_graph_artifact(target / "residue_contact_graph")
        elif representation == "cloud":
            _save_artifact(target / "residue_point_cloud")
        elif representation == "matrix":
            _save_matrix_artifact(target / "residue_contact_map")
        else:
            _save_grid_artifact(target / "residue_voxel_grid")
    index = tmp_path / "index.csv"
    _pack(root, index)

    samples = list(ErisFolderDataset(index))
    out = collate(samples)

    assert out["id"] == ["first", "second"]
    key = "x" if representation in {"graph", "matrix", "grid"} else "pos"
    assert tuple(out[key].shape) == expected_shape


def test_dataset_loads_synthetic_learned_artifact_without_gearnet_runtime(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _save_learned_artifact(root / "sample" / "synthetic_learned_embedding")
    index = tmp_path / "index.csv"
    _pack(root, index)

    item = ErisFolderDataset(index)[0]

    assert item["representation_family"] == "learned"
    assert item["representation_type"] == "embedding"
    np.testing.assert_array_equal(item["node_ids"], np.asarray(["A:1::ALA", "A:2::GLY"]))
    np.testing.assert_array_equal(item["chain_id"], np.asarray(["A", "A"]))


def test_dataset_accepts_mixed_index_without_collating(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_graph_artifact(root / "graph" / "residue_contact_graph")
    _save_matrix_artifact(root / "matrix" / "residue_contact_map")
    _save_learned_artifact(root / "learned" / "synthetic_learned_embedding")
    index = tmp_path / "index.csv"
    _pack(root, index)

    dataset = ErisFolderDataset(index)

    assert len(dataset) == 3
    assert [row.id for row in dataset.rows] == ["graph", "learned", "matrix"]
    assert dataset[1]["representation_family"] == "learned"


def test_dataset_batch_pack_dataset_gate(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    results = batch_encode(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="residue_contact_map",
        artifact_root=root,
        use_cache=False,
    )
    assert results[0].status == "success"
    index = tmp_path / "index.csv"
    pack_artifacts(root, index)

    dataset = ErisFolderDataset(index)

    assert len(dataset) == 1
    assert dataset.rows[0].id == "protein"
    assert dataset.rows[0].representation_name == "residue_contact_map"


def test_dataset_reports_actionable_error_when_torch_conversion_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    _save_artifact(root / "sample" / "residue_point_cloud")
    index = tmp_path / "index.csv"
    _pack(root, index)

    def missing_torch(*args: object, **kwargs: object) -> dict[str, object]:
        raise ImportError("No module named torch")

    import eris.datasets.torch_dataset as dataset_module

    monkeypatch.setattr(dataset_module, "to_torch", missing_torch)

    with pytest.raises(RuntimeError, match="requires optional dependency 'torch'"):
        ErisFolderDataset(index)[0]


def test_dataset_import_does_not_import_optional_frameworks() -> None:
    script = """
import sys
import eris
from eris.datasets import ErisFolderDataset

blocked = {"torch", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
assert ErisFolderDataset is not None
"""
    subprocess.run([sys.executable, "-c", script], check=True)
