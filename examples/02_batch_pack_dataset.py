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
    # Batch encode, pack and load as a dataset

    A manifest (`id`, `path`, `chain`, `model`) is encoded with
    `eris.batch_encode`, the artifacts are indexed with `eris.pack`, and the
    index is consumed by `ErisFolderDataset`.

    Equivalent CLI:

    ```bash
    eris batch encode MANIFEST --representation residue_contact_map --out ARTIFACT_ROOT
    eris pack ARTIFACT_ROOT --out index.csv
    ```
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import polars as pl

    import eris
    from eris.datasets import ErisFolderDataset

    DATA = Path(__file__).resolve().parents[1] / "tests" / "data"
    return DATA, ErisFolderDataset, Path, eris, pl, tempfile


@app.cell
def _(DATA, Path, ErisFolderDataset, eris, pl, tempfile):
    manifest = pl.DataFrame(
        {
            "id": ["1AQ2", "3EC1"],
            "path": [str(DATA / "1AQ2.pdb"), str(DATA / "3EC1.pdb")],
            "chain": ["A", "A"],
            "model": [0, 0],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "artifacts"
        results = eris.batch_encode(
            manifest,
            representation="residue_contact_map",
            artifact_root=root,
            results_path=Path(tmp) / "results.csv",
            use_cache=False,
        )
        print(results)
        index = eris.pack(root, Path(tmp) / "index.csv")
        print(index)
        dataset = ErisFolderDataset(Path(tmp) / "index.csv")
        print(len(dataset))
        sample = dataset[0]
        print(sorted(sample))
    return


if __name__ == "__main__":
    app.run()
