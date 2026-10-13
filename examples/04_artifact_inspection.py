import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Artifact inspection

    An artifact is a directory with `meta.json` and `data.npz`. Save one,
    reload it with `StructureArtifact.load`, and read its metadata.

    Equivalent CLI:

    ```bash
    eris info show DIR
    eris info validate DIR
    ```
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import eris
    from eris.core.artifacts import StructureArtifact

    PDB = Path(__file__).resolve().parents[1] / "tests" / "data" / "1AQ2.pdb"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "artifact"
        eris.encode(PDB, representation="residue_distogram", output=out, use_cache=False)
        print(sorted(p.name for p in out.iterdir()))
        loaded = StructureArtifact.load(out)
    meta = loaded.meta
    print(meta["representation_name"], meta["representation_family"], meta["schema_version"])
    print("provenance:", meta["provenance"])
    print("outputs:", [o["name"] for o in meta["outputs"]])
    print("shapes:", meta["shapes"])
    return


if __name__ == "__main__":
    app.run()
