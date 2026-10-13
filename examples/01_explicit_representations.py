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
    # Explicit representations

    Discover the available representations, encode one structure with
    `eris.encode`, and inspect the resulting `StructureArtifact`
    (`artifact.meta` / `artifact.data`). Plotting lives only in these examples.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt

    import eris

    PDB = Path(__file__).resolve().parents[1] / "tests" / "data" / "1AQ2.pdb"
    return PDB, eris, plt


@app.cell
def _(eris):
    names = eris.list_representations()
    print(names)
    info = eris.get_representation_info("residue_contact_map")
    print(info["type"], info["granularity"], info["defaults"]["cutoff"])
    return


@app.cell
def _(PDB, eris):
    artifacts = {
        name: eris.encode(PDB, representation=name, use_cache=False)
        for name in ("residue_knn_graph", "residue_contact_map", "residue_point_cloud")
    }
    for name, art in artifacts.items():
        print(name, art.meta["representation_type"], {k: v.shape for k, v in art.data.items() if v.ndim > 1})
    return (artifacts,)


@app.cell
def _(artifacts, plt):
    contact = artifacts["residue_contact_map"].data["contact"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(contact, cmap="Greys")
    ax.set_title("1AQ2 CA contact map (8 A)")
    ax.set_xlabel("residue")
    ax.set_ylabel("residue")
    fig
    return


if __name__ == "__main__":
    app.run()
