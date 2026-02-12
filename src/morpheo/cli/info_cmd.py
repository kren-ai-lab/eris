from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Artifact inspection utilities.")


@app.command()
def show(
    artifact_dir: Path = typer.Argument(..., exists=True, readable=True),
    strict: bool = typer.Option(False, "--strict", help="Strictly enforce shapes/dtypes match data.npz"),
) -> None:
    """
    Print artifact meta and validate schema.
    """
    from morpheo.core.artifacts import StructureArtifact

    art = StructureArtifact.load(artifact_dir, validate=True, strict=strict)

    meta = art.meta
    typer.echo(f"OK | schema_version={meta.get('schema_version')}")
    typer.echo(f"encoder: {meta.get('encoder_name')} | version={meta.get('encoder_version')}")
    typer.echo(f"type={meta.get('type')} level={meta.get('level')} family={meta.get('family')}")
    typer.echo(f"input: {meta.get('input')}")
    typer.echo(f"counts: {meta.get('counts')}")
    typer.echo(f"keys: {list(art.data.keys())}")

    typer.echo("shapes:")
    for k, shp in (meta.get('shapes') or {}).items():
        typer.echo(f"  - {k}: {shp}")
