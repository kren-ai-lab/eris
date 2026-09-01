from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((ROOT / "notebooks").glob("*.ipynb")))
OBSOLETE_EXAMPLE_TEXT = (
    "eris encode batch",
    "eris embed batch",
    "eris encode run",
    "eris export batch-to-torch",
    "eris export batch-to-pyg",
    "from eris.core.registry",
    "from eris.core.types",
    "get_encoder(",
    "get_encoder_info(",
    "../out",
    "..\\out",
)


def _notebook(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_tracked_notebooks_are_source_only_and_use_current_boundaries() -> None:
    assert NOTEBOOKS

    for path in NOTEBOOKS:
        notebook = _notebook(path)
        assert notebook["nbformat"] == 4
        text = "\n".join(_source(cell) for cell in notebook["cells"])
        normalized = text.casefold()
        for obsolete in OBSOLETE_EXAMPLE_TEXT:
            assert obsolete not in normalized, f"{path.name}: {obsolete}"

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, f"{path.name}: cell {index}"
            assert cell.get("outputs") == [], f"{path.name}: cell {index}"
            compile(_source(cell), f"{path.name}: cell {index}", "exec")


def test_historical_backlog_is_not_presented_as_current_v01_scope() -> None:
    backlog = (ROOT / "TODO.md").read_text(encoding="utf-8")

    assert backlog.startswith("# Historical post-v0.1 backlog and reference checklist")
    assert "docs/milestones.md` is authoritative" in backlog
    assert "current v0.1 does not provide `eris[plm]`" in backlog
    assert "pip install eris[plm]" not in backlog
