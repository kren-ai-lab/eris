import subprocess
import sys
from types import MappingProxyType

import pytest

import eris
from eris.cli.main import EXIT_CONFIG, EXIT_WORKFLOW, main


def test_valid_mapping_and_path(tmp_path):
    config = {"representation": "residue_knn_graph", "params": {"k": 4}, "chain": "A", "model": 0}
    assert eris.validate_config(config, "encode") == []
    assert eris.validate_config(MappingProxyType(config), "encode") == []
    path = tmp_path / "cfg.yaml"
    path.write_text("representation: residue_knn_graph\nmodel: 1\n", encoding="utf-8")
    assert eris.validate_config(path, "encode") == []
    assert eris.validate_config(str(path), "encode") == []


def test_invalid_mapping_reports_dotted_paths():
    config = {"representation": "residue_knn_graph", "params": {"nope": 1}, "bad": 1, "chain": 5}
    issues = {issue.path: issue.message for issue in eris.validate_config(config, "encode")}
    assert issues["params.nope"] == "unknown parameter"
    assert issues["bad"] == "unknown key"
    assert issues["chain"] == "expected a string, got int"


def test_model_is_the_config_key():
    assert eris.validate_config({"model": 1}, "embed") == []
    assert [i.path for i in eris.validate_config({"model_index": 1}, "embed")] == ["model_index"]


def test_missing_or_unparsable_file_is_one_root_issue(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [", encoding="utf-8")
    for path in (tmp_path / "missing.yaml", bad):
        issues = eris.validate_config(path, "encode")
        assert [issue.path for issue in issues] == [""]


def test_unknown_target_is_one_root_issue():
    assert [issue.path for issue in eris.validate_config({}, "nope")] == [""]


def test_validate_config_does_not_import_torch():
    code = (
        "import sys, eris; eris.validate_config({'chain': 'A'}, 'embed'); "
        "sys.exit(any(m.split('.')[0] in ('torch', 'torch_geometric') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603


@pytest.mark.parametrize(
    "args",
    [
        ["batch", "encode", "missing.csv"],
        ["batch", "embed", "missing.csv"],
        ["batch", "from-folder", "missing-dir"],
    ],
)
def test_missing_batch_input_exits_3(args, tmp_path):
    code = main([*args, "-r", "residue_knn_graph", "--out", str(tmp_path / "out")])
    assert code == EXIT_WORKFLOW


def test_missing_config_file_exits_2(tmp_path):
    code = main(["batch", "encode", "m.csv", "-r", "x", "--out", str(tmp_path), "--config", "nope.yaml"])
    assert code == EXIT_CONFIG
