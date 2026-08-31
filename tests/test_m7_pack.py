from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from eris.cli.main import app
from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import StructureArtifact
from eris.pipelines.batch_encode import batch_encode
from eris.pipelines.manifest import StructureManifestRow
from eris.pipelines.pack import (
    PACKED_INDEX_FIELDS,
    PackedIndexRow,
    PackIndexError,
    pack_artifacts,
)

PDB = Path("tests/data/1AQ2.pdb")
pack_module = importlib.import_module("eris.pipelines.pack")


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _resolve_portable(anchor: Path, value: str) -> Path:
    return (anchor / Path(*value.split("/"))).resolve()


def _assert_row_paths_resolve(index_path: Path, row: dict[str, str], artifact_dir: Path) -> None:
    assert _resolve_portable(index_path.parent, row["artifact_dir"]) == artifact_dir.resolve()
    assert (
        _resolve_portable(index_path.parent, row["meta_path"])
        == (artifact_dir / "meta.json").resolve()
    )
    assert not Path(row["artifact_dir"]).is_absolute()
    assert not Path(row["meta_path"]).is_absolute()
    assert "\\" not in row["artifact_dir"]
    assert "\\" not in row["meta_path"]


def _save_explicit_artifact(
    target: Path,
    *,
    representation_name: str = "residue_point_cloud",
    representation_type: str = "cloud",
    data: dict[str, np.ndarray] | None = None,
    outputs: list[dict[str, object]] | None = None,
    counts: dict[str, int] | None = None,
) -> StructureArtifact:
    if data is None:
        data = {"pos": np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)}
    if outputs is None:
        outputs = [
            {
                "name": next(iter(data)),
                "role": "coordinates",
                "granularity": "residue",
                "layout": "NC",
            }
        ]
    parameters = {"fixture": representation_name}
    metadata = build_artifact_metadata(
        data=data,
        representation_name=representation_name,
        representation_family="explicit",
        representation_type=representation_type,
        granularity="residue",
        coordinate_source="ca",
        encoder={
            "implementation": "tests.fixtures.PackExplicitProducer",
            "version": "1",
        },
        input_metadata=build_input_metadata(
            source="memory://explicit",
            source_sha256="c" * 64,
            input_format="pdb",
            model_index=0,
            chain_id=None,
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name=representation_name,
            parameters=parameters,
        ),
        output_declarations=outputs,
        feature_keys=build_feature_keys(),
        counts={} if counts is None else counts,
    )
    artifact = StructureArtifact(data=data, meta=metadata)
    artifact.save(target)
    return artifact


def _save_learned_artifact(target: Path) -> StructureArtifact:
    data = {
        "node_embedding": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "global_embedding": np.asarray([0.5, 0.6], dtype=np.float32),
        "node_ids": np.asarray(["A:1::ALA", "A:2::GLY"], dtype=np.str_),
    }
    parameters = {"output_mode": "both"}
    metadata = build_artifact_metadata(
        data=data,
        representation_name="synthetic_learned_embedding",
        representation_family="learned",
        representation_type="embedding",
        granularity="mixed",
        coordinate_source="model_specific",
        encoder={
            "implementation": "tests.fixtures.PackLearnedAdapter",
            "version": "1",
        },
        input_metadata=build_input_metadata(
            source="memory://learned",
            source_sha256="d" * 64,
            input_format="pdb",
            model_index=0,
            chain_id="A",
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="synthetic_learned_embedding",
            parameters=parameters,
        ),
        output_declarations=[
            {"name": "node_embedding", "role": "node_embedding", "layout": "ND"},
            {"name": "global_embedding", "role": "global_embedding", "layout": "D"},
            {
                "name": "node_ids",
                "role": "node_ids",
                "granularity": "residue",
                "layout": "N",
            },
        ],
        feature_keys=build_feature_keys(),
        counts={"n_nodes": 2},
        model={
            "architecture": "SyntheticModel",
            "checkpoint": {
                "identifier": "synthetic.pth",
                "revision": "test-revision",
                "source": "memory://checkpoint",
                "sha256": "e" * 64,
            },
        },
    )
    artifact = StructureArtifact(data=data, meta=metadata)
    artifact.save(target)
    return artifact


def test_packed_index_row_schema_is_exact() -> None:
    row = PackedIndexRow(
        id="protein",
        artifact_dir="protein/residue_point_cloud",
        meta_path="protein/residue_point_cloud/meta.json",
        representation_name="residue_point_cloud",
        representation_family="explicit",
        representation_type="cloud",
        granularity="residue",
        n_nodes=1,
        n_edges=None,
        shape_summary='{"pos":[1,3]}',
    )

    assert tuple(row.to_dict()) == PACKED_INDEX_FIELDS


def test_pack_empty_root_writes_header_only_index(tmp_path: Path) -> None:
    out = tmp_path / "index.csv"

    rows = pack_artifacts(tmp_path, out)

    assert rows == []
    assert out.read_text(encoding="utf-8").splitlines() == [",".join(PACKED_INDEX_FIELDS)]


def test_pack_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(PackIndexError, match="does not exist"):
        pack_artifacts(tmp_path / "missing", tmp_path / "index.csv")


def test_pack_rejects_non_directory_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "not-a-dir"
    artifact_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(PackIndexError, match="not a directory"):
        pack_artifacts(artifact_root, tmp_path / "index.csv")


def test_pack_one_explicit_artifact_uses_relative_portable_paths_and_metadata(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "protein_001" / "residue_point_cloud"
    _save_explicit_artifact(artifact_dir, counts={"n_nodes": 1})
    out = tmp_path / "index.csv"

    rows = pack_artifacts(tmp_path / "artifacts", out)
    payload = _read_index(out)

    assert len(rows) == 1
    assert payload == [
        {
            "id": "protein_001",
            "artifact_dir": "artifacts/protein_001/residue_point_cloud",
            "meta_path": "artifacts/protein_001/residue_point_cloud/meta.json",
            "representation_name": "residue_point_cloud",
            "representation_family": "explicit",
            "representation_type": "cloud",
            "granularity": "residue",
            "n_nodes": "1",
            "n_edges": "",
            "shape_summary": '{"pos":[1,3]}',
        }
    ]
    _assert_row_paths_resolve(out, payload[0], artifact_dir)


def test_pack_supports_single_artifact_root_without_source_filename_identity(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "single-artifact"
    _save_explicit_artifact(artifact_dir, counts={"n_nodes": 1})
    out = tmp_path / "index.csv"

    rows = pack_artifacts(artifact_dir, out)

    assert rows[0].id == "single-artifact"
    assert rows[0].artifact_dir == "single-artifact"
    assert rows[0].meta_path == "single-artifact/meta.json"


def test_pack_paths_are_relative_to_index_parent_for_sibling_output(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "sample" / "residue_point_cloud"
    _save_explicit_artifact(artifact_dir)
    index_path = tmp_path / "index.csv"

    pack_artifacts(tmp_path / "artifacts", index_path)

    row = _read_index(index_path)[0]
    assert row["artifact_dir"] == "artifacts/sample/residue_point_cloud"
    assert row["meta_path"] == "artifacts/sample/residue_point_cloud/meta.json"
    _assert_row_paths_resolve(index_path, row, artifact_dir)


def test_pack_paths_are_relative_to_index_parent_when_output_is_inside_root(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "sample" / "residue_point_cloud"
    _save_explicit_artifact(artifact_dir)
    index_path = tmp_path / "artifacts" / "index.csv"

    pack_artifacts(tmp_path / "artifacts", index_path)

    row = _read_index(index_path)[0]
    assert row["artifact_dir"] == "sample/residue_point_cloud"
    assert row["meta_path"] == "sample/residue_point_cloud/meta.json"
    _assert_row_paths_resolve(index_path, row, artifact_dir)


def test_pack_paths_are_relative_to_index_parent_when_output_is_elsewhere(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "sample" / "residue_point_cloud"
    _save_explicit_artifact(artifact_dir)
    index_path = tmp_path / "collections" / "index.csv"

    pack_artifacts(tmp_path / "artifacts", index_path)

    row = _read_index(index_path)[0]
    assert row["artifact_dir"] == "../artifacts/sample/residue_point_cloud"
    assert row["meta_path"] == "../artifacts/sample/residue_point_cloud/meta.json"
    _assert_row_paths_resolve(index_path, row, artifact_dir)


def test_pack_multiple_artifacts_recursive_discovery_and_deterministic_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _save_explicit_artifact(root / "b" / "residue_point_cloud")
    _save_explicit_artifact(root / "a" / "residue_point_cloud")
    _save_explicit_artifact(root / "nested" / "c" / "residue_point_cloud")
    out = tmp_path / "index.csv"

    rows = pack_artifacts(root, out)

    assert [row.artifact_dir for row in rows] == [
        "artifacts/a/residue_point_cloud",
        "artifacts/b/residue_point_cloud",
        "artifacts/nested/c/residue_point_cloud",
    ]
    assert [row.id for row in rows] == ["a", "b", "nested/c"]


def test_pack_handles_mixed_explicit_and_learned_collection(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_explicit_artifact(root / "a" / "residue_point_cloud", counts={"n_nodes": 1})
    _save_explicit_artifact(
        root / "b" / "residue_contact_map",
        representation_name="residue_contact_map",
        representation_type="matrix",
        data={"contact": np.asarray([[1]], dtype=np.bool_)},
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
    _save_learned_artifact(root / "c" / "synthetic_learned_embedding")
    out = tmp_path / "index.csv"

    rows = pack_artifacts(root, out)

    assert [(row.id, row.representation_family, row.representation_name) for row in rows] == [
        ("a", "explicit", "residue_point_cloud"),
        ("b", "explicit", "residue_contact_map"),
        ("c", "learned", "synthetic_learned_embedding"),
    ]


def test_pack_uses_representation_metadata_not_path_name(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "sample" / "misleading_directory"
    _save_explicit_artifact(
        artifact_dir,
        representation_name="residue_point_cloud",
        counts={"n_nodes": 1},
    )
    out = tmp_path / "index.csv"

    rows = pack_artifacts(tmp_path / "artifacts", out)

    assert rows[0].id == "sample/misleading_directory"
    assert rows[0].representation_name == "residue_point_cloud"


def test_pack_uses_counts_without_inventing_non_applicable_edges(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_explicit_artifact(root / "cloud" / "residue_point_cloud", counts={"n_nodes": 1})
    _save_explicit_artifact(
        root / "graph" / "residue_contact_graph",
        representation_name="residue_contact_graph",
        representation_type="graph",
        data={
            "node_attr": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        },
        outputs=[
            {
                "name": "node_attr",
                "role": "node_features",
                "granularity": "residue",
                "layout": "NF",
            },
            {"name": "edge_index", "role": "edge_index", "layout": "2E"},
        ],
        counts={"n_nodes": 2, "n_edges": 2},
    )
    out = tmp_path / "index.csv"

    rows = pack_artifacts(root, out)

    assert [(row.id, row.n_nodes, row.n_edges) for row in rows] == [
        ("cloud", 1, None),
        ("graph", 2, 2),
    ]


def test_pack_shape_summary_is_compact_sorted_json(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_explicit_artifact(
        root / "protein" / "custom_matrix",
        representation_name="custom_matrix",
        representation_type="matrix",
        data={
            "zeta": np.zeros((2, 2), dtype=np.float32),
            "alpha": np.zeros((2,), dtype=np.float32),
        },
        outputs=[
            {"name": "zeta", "role": "distance_matrix", "granularity": "residue", "layout": "NN"},
            {"name": "alpha", "role": "metadata", "layout": "N"},
        ],
        counts={"n_nodes": 2},
    )
    out = tmp_path / "index.csv"

    rows = pack_artifacts(root, out)

    assert rows[0].shape_summary == '{"alpha":[2],"zeta":[2,2]}'


def test_pack_rejects_schema_v1_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "legacy" / "residue_point_cloud"
    artifact_dir.mkdir(parents=True)
    np.savez_compressed(artifact_dir / "data.npz", pos=np.zeros((1, 3), dtype=np.float32))
    (artifact_dir / "meta.json").write_text(
        json.dumps({"schema_version": 1, "encoder_name": "legacy"}),
        encoding="utf-8",
    )

    with pytest.raises(PackIndexError, match="Invalid Artifact Schema v2"):
        pack_artifacts(tmp_path / "artifacts", tmp_path / "index.csv")


def test_pack_rejects_corrupt_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "corrupt" / "residue_point_cloud"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "meta.json").write_text("{bad json", encoding="utf-8")
    (artifact_dir / "data.npz").write_bytes(b"not npz")

    with pytest.raises(PackIndexError, match="Invalid Artifact Schema v2"):
        pack_artifacts(tmp_path / "artifacts", tmp_path / "index.csv")


def test_pack_rejects_incomplete_artifact_candidates(tmp_path: Path) -> None:
    partial = tmp_path / "artifacts" / "partial"
    partial.mkdir(parents=True)
    (partial / "meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PackIndexError, match="Incomplete artifact directories"):
        pack_artifacts(tmp_path / "artifacts", tmp_path / "index.csv")


def test_pack_rejects_duplicate_id_representation_ambiguity(tmp_path: Path) -> None:
    rows = [
        PackedIndexRow(
            id="sample",
            artifact_dir="sample/residue_point_cloud",
            meta_path="sample/residue_point_cloud/meta.json",
            representation_name="residue_point_cloud",
            representation_family="explicit",
            representation_type="cloud",
            granularity="residue",
            n_nodes=1,
            n_edges=None,
            shape_summary='{"pos":[1,3]}',
        ),
        PackedIndexRow(
            id="sample",
            artifact_dir="sample/other",
            meta_path="sample/other/meta.json",
            representation_name="residue_point_cloud",
            representation_family="explicit",
            representation_type="cloud",
            granularity="residue",
            n_nodes=1,
            n_edges=None,
            shape_summary='{"pos":[1,3]}',
        ),
    ]

    with pytest.raises(PackIndexError, match="duplicate artifact"):
        pack_module._validate_index_rows(rows, tmp_path)


def test_pack_does_not_follow_symlink_directories(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    external = tmp_path / "external" / "sample" / "residue_point_cloud"
    _save_explicit_artifact(external)
    link = root / "linked"
    root.mkdir()
    try:
        link.symlink_to(tmp_path / "external", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")
    out = tmp_path / "index.csv"

    rows = pack_artifacts(root, out)

    assert rows == []


def test_pack_preserves_existing_output_on_validation_failure(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    partial = root / "partial"
    partial.mkdir(parents=True)
    (partial / "meta.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "index.csv"
    original = "previous,index\nstill,valid\n"
    out.write_text(original, encoding="utf-8")

    with pytest.raises(PackIndexError):
        pack_artifacts(root, out)

    assert out.read_text(encoding="utf-8") == original


def test_pack_is_metadata_only_and_does_not_rewrite_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_explicit_artifact(root / "sample" / "residue_point_cloud")
    before = _file_bytes(root)

    pack_artifacts(root, tmp_path / "index.csv")

    assert _file_bytes(root) == before


def test_cli_pack_help_exposes_artifact_root_and_out() -> None:
    result = CliRunner().invoke(app, ["pack", "--help"])

    assert result.exit_code == 0
    assert "ARTIFACT_ROOT" in result.output
    assert "--out" in result.output
    assert "--compress" not in result.output
    assert "--stack" not in result.output
    assert "--pad" not in result.output
    assert "--recursive" not in result.output


def test_cli_pack_execution_and_clean_error(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _save_explicit_artifact(root / "sample" / "residue_point_cloud")
    out = tmp_path / "index.csv"
    runner = CliRunner()

    result = runner.invoke(app, ["pack", str(root), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "Packed 1 artifacts" in result.output
    assert _read_index(out)[0]["artifact_dir"] == "artifacts/sample/residue_point_cloud"

    missing = runner.invoke(app, ["pack", str(tmp_path / "missing"), "--out", str(out)])
    assert missing.exit_code == 2
    assert "does not exist" in missing.output
    assert "Traceback" not in missing.output


def test_pack_does_not_import_optional_frameworks_in_subprocess(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import eris
from eris.pipelines.pack import pack_artifacts

with TemporaryDirectory() as tmpdir:
    root = Path(tmpdir) / "artifacts"
    artifact_dir = root / "sample" / "residue_point_cloud"
    eris.encode(
        "tests/data/1AQ2.pdb",
        representation="residue_point_cloud",
        output=artifact_dir,
        use_cache=False,
    )
    pack_artifacts(root, Path(tmpdir) / "index.csv")

blocked = {"torch", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", script], check=True)


def test_m7_2_batch_output_can_be_packed_without_batch_results_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    results = batch_encode(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="residue_contact_map",
        artifact_root=root,
        use_cache=False,
    )
    assert results[0].status == "success"

    rows = pack_artifacts(root, tmp_path / "index.csv")

    assert [(row.id, row.representation_name) for row in rows] == [
        ("protein", "residue_contact_map")
    ]
    assert rows[0].artifact_dir == "artifacts/protein/residue_contact_map"
