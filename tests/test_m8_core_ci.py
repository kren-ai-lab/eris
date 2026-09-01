from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_core_ci_matrix_enforces_the_frozen_core_contract() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    core = workflow.split("  test:\n", maxsplit=1)[1].split("  distribution-build:\n", maxsplit=1)[
        0
    ]
    normalized = core.casefold()

    assert 'python-version: ["3.10", "3.11", "3.12", "3.13"]' in core
    assert "permissions:\n  contents: read" in workflow
    assert 'python -m pip install -e ".[dev]"' in core
    assert "python -m ruff check ." in workflow
    assert "python -m ruff format --check ." in workflow
    assert "python -m pytest -q" in workflow
    assert "python -m pip check" in workflow

    for forbidden in (".[gearnet]", "torchdrug", "cuda", "gpu"):
        assert forbidden not in normalized
