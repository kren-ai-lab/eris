# Eris examples

Runnable [marimo](https://marimo.io) notebooks showing complete `eris`
workflows on the structures in `tests/data`. Plotting lives only here, never in
the `eris` package. GearNet examples are not included because they need an
official checkpoint download.

## Catalog

| Example | API used | What it shows |
|---|---|---|
| [01_explicit_representations.py](01_explicit_representations.py) | `list_representations`, `get_representation_info`, `encode` | discovery, graph / contact map / point cloud artifacts, `meta` and `data` shapes, contact-map plot |
| [02_batch_pack_dataset.py](02_batch_pack_dataset.py) | `batch_encode`, `pack`, `ErisFolderDataset` | Polars manifest, batch results, packed index and dataset loading (plus CLI equivalents) |
| [03_torch_pyg_export.py](03_torch_pyg_export.py) | `to_torch`, `to_pyg` | tensor export and PyG export of a graph artifact with public topology |
| [04_artifact_inspection.py](04_artifact_inspection.py) | `StructureArtifact.load` | saving, reloading and inspecting meta, provenance and outputs (plus `eris info`) |

## Running

Each example is a marimo notebook stored as a plain Python file. Run it as a
script, or open it interactively:

```bash
uv sync --all-extras --group examples
MPLBACKEND=Agg uv run python examples/01_explicit_representations.py   # script
uv run marimo edit examples/01_explicit_representations.py             # notebook
```

Run the complete suite with `bash examples/run_ci_examples.sh`. All outputs go
to temporary directories.
