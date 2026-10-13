from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import eris
from eris.cli.main import EXIT_OK, main
from eris.export.table import FeatureTableError
from tests.helpers import MULTIMER_PDB, PDB, learned_artifact


def _graph(path: Path) -> eris.core.artifacts.StructureArtifact:
    return eris.encode(path, representation="residue_knn_graph", chain_id="A", use_cache=False)


def test_pooled_artifacts_become_named_feature_columns() -> None:
    first, second = eris.pool(_graph(PDB)), eris.pool(_graph(MULTIMER_PDB))

    table = eris.feature_table({"p1": first, "p2": second})

    keys = first.meta["feature_keys"]["pooled_attr_keys"]
    assert table.columns == ["id", *keys]
    assert table["id"].to_list() == ["p1", "p2"]
    np.testing.assert_array_equal(
        table.drop("id").to_numpy(), np.vstack([first.data["pooled_attr"], second.data["pooled_attr"]])
    )


def test_learned_global_embeddings_keep_dtype_and_get_positional_names() -> None:
    table = eris.feature_table({"a": learned_artifact("both"), "b": learned_artifact("both")})

    assert table.columns == ["id", "p_0", "p_1", "p_2"]
    assert table.schema["p_0"] == pl.Float32
    np.testing.assert_array_equal(table.drop("id").to_numpy(), np.asarray([[7, 8, 9], [7, 8, 9]], np.float32))


def test_packed_index_rows_use_index_ids_and_explicit_source(tmp_path: Path) -> None:
    manifest = pl.DataFrame({"id": ["p1", "p2"], "path": [str(PDB), str(MULTIMER_PDB)]})
    eris.batch_encode(
        manifest, representation="residue_knn_graph", artifact_root=tmp_path / "a", use_cache=False
    )
    index = tmp_path / "index.csv"
    eris.pack(tmp_path / "a", index)

    table = eris.feature_table(index, source="graph_attr")

    assert table["id"].to_list() == ["p1", "p2"]
    assert "density" in table.columns
    assert table.height == 2


def test_nan_values_are_copied_not_replaced() -> None:
    artifact = eris.pool(_graph(PDB), source="node_attr", mode="mean")
    artifact.data["pooled_attr"][0] = np.nan

    table = eris.feature_table({"p1": artifact})

    assert np.isnan(table.row(0)[1])


def test_artifacts_without_a_structure_vector_are_rejected() -> None:
    contact_map = eris.encode(PDB, representation="residue_contact_map", use_cache=False)

    with pytest.raises(FeatureTableError, match=r"no structure-level vector.*eris\.pool"):
        eris.feature_table({"p1": contact_map})


def test_rows_with_different_configurations_are_rejected() -> None:
    graph = _graph(PDB)

    with pytest.raises(FeatureTableError, match=r"p2: differs from p1 in parameters"):
        eris.feature_table({"p1": eris.pool(graph, mode="mean"), "p2": eris.pool(graph, mode="max")})


def test_cli_features_writes_the_same_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = pl.DataFrame({"id": ["p1"], "path": [str(PDB)]})
    eris.batch_encode(
        manifest, representation="residue_knn_graph", artifact_root=tmp_path / "a", use_cache=False
    )
    index = tmp_path / "index.csv"
    eris.pack(tmp_path / "a", index)
    out = tmp_path / "features.csv"

    code = main(["features", str(index), "--source", "graph_attr", "--out", str(out)])

    assert code == EXIT_OK
    assert "Wrote 1 rows" in capsys.readouterr().out
    assert pl.read_csv(out).columns == eris.feature_table(index, source="graph_attr").columns
