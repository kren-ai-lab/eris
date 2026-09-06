from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.cli._utils import usage_error

app = typer.Typer(add_completion=False, help="Artifact inspection utilities.")
console = Console(width=240)


def _load_artifact(artifact_dir: Path) -> StructureArtifact:
    try:
        return StructureArtifact.load(artifact_dir, strict=True)
    except ArtifactValidationError as error:
        usage_error(str(error))


def _json(value: Any) -> Syntax:
    return Syntax(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        "json",
        word_wrap=True,
    )


def _section(title: str, rows: list[tuple[str, Any]]) -> None:
    table = Table(title=title, show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", no_wrap=True)
    for key, value in rows:
        if isinstance(value, (dict, list)):
            table.add_row(key, _json(value))
        else:
            table.add_row(key, "" if value is None else str(value))
    console.print(table)


def _model_rows(model: dict[str, Any]) -> list[tuple[str, Any]]:
    checkpoint = model.get("checkpoint")
    rows: list[tuple[str, Any]] = [("architecture", model.get("architecture"))]
    if isinstance(checkpoint, dict):
        rows.extend(
            [
                ("checkpoint identifier", checkpoint.get("identifier")),
                ("checkpoint revision", checkpoint.get("revision")),
                ("checkpoint source", checkpoint.get("source")),
                ("checkpoint sha256", checkpoint.get("sha256")),
            ]
        )
    return rows


@app.command()
def show(
    artifact_dir: Annotated[Path, typer.Argument(...)],
) -> None:
    """
    Print canonical Artifact Schema metadata after strict validation.
    """

    artifact = _load_artifact(artifact_dir)
    meta = artifact.meta
    encoder = meta.get("encoder")

    _section(
        "Artifact",
        [
            ("schema_version", meta.get("schema_version")),
            ("representation_name", meta.get("representation_name")),
            ("representation_family", meta.get("representation_family")),
            ("representation_type", meta.get("representation_type")),
            ("granularity", meta.get("granularity")),
            ("coordinate_source", meta.get("coordinate_source")),
        ],
    )
    _section("Input", [("input", meta.get("input"))])
    if isinstance(encoder, dict):
        _section(
            "Encoder",
            [
                ("implementation", encoder.get("implementation")),
                ("version", encoder.get("version")),
            ],
        )
    model = meta.get("model")
    if isinstance(model, dict):
        _section("Model", _model_rows(model))
    _section(
        "Metadata",
        [
            ("parameters", meta.get("parameters")),
            ("outputs", meta.get("outputs")),
            ("feature_keys", meta.get("feature_keys")),
            ("shapes", meta.get("shapes")),
            ("dtypes", meta.get("dtypes")),
            ("counts", meta.get("counts")),
            ("provenance", meta.get("provenance")),
        ],
    )
    _section("Data", [("keys", list(artifact.data))])


@app.command()
def validate(
    artifact_dir: Annotated[Path, typer.Argument(...)],
) -> None:
    """
    Validate a canonical Artifact Schema directory.
    """

    artifact = _load_artifact(artifact_dir)
    typer.echo(f"Artifact is valid: {artifact.meta['representation_name']}")
