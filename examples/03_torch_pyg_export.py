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
    # Torch and PyG export

    `eris.to_torch` converts any artifact to tensors. `eris.to_pyg` only
    accepts explicit graph artifacts, because it uses only the topology that
    is already public in the artifact. `to_torch` needs PyTorch (any of the
    `gearnet` or `pyg` extras); `to_pyg` needs the `pyg` extra.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import eris

    PDB = Path(__file__).resolve().parents[1] / "tests" / "data" / "1AQ2.pdb"
    graph = eris.encode(PDB, representation="residue_knn_graph", use_cache=False)
    return eris, graph


@app.cell
def _(eris, graph):
    tensors = eris.to_torch(graph)
    print({k: (tuple(v.shape), str(v.dtype)) for k, v in tensors.items() if hasattr(v, "shape")})
    return


@app.cell
def _(eris, graph):
    data = eris.to_pyg(graph)
    print(data)
    return


if __name__ == "__main__":
    app.run()
