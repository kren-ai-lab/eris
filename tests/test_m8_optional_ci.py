from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_distribution_and_optional_ci_keep_their_frozen_boundaries() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    normalized = workflow.casefold()
    distribution = workflow.split("  distribution-build:\n", maxsplit=1)[1].split(
        "  learned-gearnet:\n", maxsplit=1
    )[0]
    learned = workflow.split("  learned-gearnet:\n", maxsplit=1)[1]

    assert "python -m build --outdir dist" in distribution
    assert 'python-version: ["3.12", "3.13"]' in distribution
    assert "python-version: ${{ matrix.python-version }}" in distribution
    assert "tests/test_m8_release_boundaries.py" in distribution
    assert "python -m venv /tmp/eris-wheel-env" in distribution
    assert "pip install dist/*.whl" in distribution
    assert "eris.__version__ == version('eris')" in distribution
    assert "pip check" in distribution

    assert 'python-version: ["3.10", "3.11", "3.12", "3.13"]' in learned
    assert "python-version: ${{ matrix.python-version }}" in learned
    assert "https://download.pytorch.org/whl/cpu" in learned
    assert 'python -m pip install -e ".[dev,gearnet]"' in learned
    assert "tests/test_gearnet_dependencies.py" in learned
    assert "tests/test_gearnet_native_graph.py" in learned
    assert "tests/test_gearnet_native_model.py" in learned
    assert "tests/test_gearnet_inference.py" in learned

    for legacy_dependency in (
        "torchdrug",
        "rdkit",
        "torch-scatter",
        "torch-cluster",
        "torch-sparse",
        "torch-geometric",
        "ninja",
    ):
        assert legacy_dependency not in learned.casefold()

    for forbidden in (
        "cuda",
        "gpu",
        "eris_gearnet_checkpoint",
        "curl ",
        "wget ",
        "publish",
        "twine",
    ):
        assert forbidden not in normalized
