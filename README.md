# Eris

Eris converts protein 3D structures into numerical representations for machine learning:
point clouds, graphs, contact maps, distograms, voxel grids, and GearNet-Edge embeddings.

- Explicit representations - structural arrays derived from PDB/mmCIF coordinates.
- Learned representations - residue-level and structure-level GearNet-Edge embeddings.
- Artifacts - NumPy arrays stored with parameters and provenance.

## Installation

Eris supports Python 3.10-3.13 (`>=3.10,<3.14`). Install from a repository checkout or source
distribution:

```console
python -m pip install .
```

For GearNet-Edge embeddings:

```console
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[gearnet]"
```

For PyTorch Geometric export:

```console
python -m pip install ".[pyg]"
```

For local development:

```console
python -m pip install ".[dev]"
```

## Quick start

Encode a structure as a residue k-nearest-neighbor graph, then generate a GearNet-Edge embedding:

```python
import eris

graph = eris.encode(
    "protein.pdb",
    representation="residue_knn_graph",
    k=8,
    output="artifact_knn",
)

embedding = eris.embed(
    "protein.pdb",
    representation="gearnet_edge",
    output_mode="both",
    checkpoint_path="mc_gearnet_edge.pth",
    output="artifact_gearnet",
)
```

`encode()` accepts representation parameters as keyword arguments. `embed()` supports `node`,
`global`, and `both` output modes.

## Structural representations

Eris provides ten explicit structural representations:

| Name | Type | Granularity | Description |
| --- | --- | --- | --- |
| `atom_point_cloud` | cloud | atom | Atomic coordinates and features |
| `residue_point_cloud` | cloud | residue | CA coordinates and residue features |
| `atom_knn_graph` | graph | atom | Atomic k-nearest-neighbor graph |
| `residue_knn_graph` | graph | residue | Residue k-nearest-neighbor graph |
| `residue_contact_graph` | graph | residue | Residue graph defined by spatial contacts |
| `residue_distogram` | matrix | residue | Pairwise CA distances |
| `residue_distogram_bins` | matrix | residue | Discretized pairwise CA distances |
| `residue_contact_map` | matrix | residue | Binary CA contact map |
| `residue_contact_map_bins` | matrix | residue | Distance-binned contact map |
| `residue_voxel_grid` | grid | residue | Multi-channel 3D residue grid |

Use `eris representations list` or `eris.get_representation_info(name)` to inspect parameters.

## GearNet-Edge embeddings

`gearnet_edge` uses the pretrained GearNet-Edge checkpoint and produces 3072-dimensional embeddings:

- `node`: one `(N, 3072)` residue embedding array;
- `global`: one `(3072,)` structure embedding;
- `both`: both outputs.

Runtime: Python 3.10-3.13, `torch==2.11.0`, CPU, and fp32. The checkpoint is not bundled or
downloaded automatically. Supply `mc_gearnet_edge.pth` through `checkpoint_path` or
`--checkpoint`.

Inspect the registered embedder and model metadata without running inference:

```console
eris embedders list
eris embedders info gearnet_edge
eris models list
eris models info gearnet_edge
```

## Command-line interface

```console
eris --help
```

| Command | Purpose |
| --- | --- |
| `encode` | Write one explicit representation as a Schema v2 artifact. |
| `embed` | Write one learned representation as a Schema v2 artifact. |
| `batch encode` | Encode one representation for each manifest row. |
| `batch embed` | Embed one representation for each manifest row. |
| `pack` | Validate artifacts and write a metadata-only index. |
| `info show` | Print artifact metadata. |
| `info validate` | Validate an artifact directory. |
| `representations list/info` | List or inspect explicit representations. |
| `embedders list/info` | List or inspect learned embedders. |
| `models list/info` | List or inspect learned model assets. |
| `export to-torch/to-pyg` | Export one artifact to Torch or PyG. |
| `export batch-to-torch/batch-to-pyg` | Export multiple artifacts to Torch or PyG. |
| `cache info/clear` | Inspect or clear Eris cache namespaces. |

Examples:

```console
eris representations list
eris representations info residue_knn_graph
eris encode protein.pdb --representation residue_knn_graph --params '{"k": 8}' --out artifact_knn
eris embed protein.pdb --representation gearnet_edge --output-mode both --checkpoint mc_gearnet_edge.pth --device cpu --precision fp32 --out artifact_gearnet
eris info validate artifact_knn
eris info show artifact_gearnet
eris export to-torch --artifact-dir artifact_knn --out graph.pt
eris export to-pyg --artifact-dir artifact_knn --out graph_pyg.pt
```

## Notebooks

| Notebook | Covers |
| --- | --- |
| [`encoders.ipynb`](notebooks/encoders.ipynb) | Explicit structural representations |
| [`batch_encode.ipynb`](notebooks/batch_encode.ipynb) | Batch processing |
| [`export.ipynb`](notebooks/export.ipynb) | Torch and PyG export |
| [`info.ipynb`](notebooks/info.ipynb) | Artifact inspection |
| [`cache.ipynb`](notebooks/cache.ipynb) | Cache inspection and clearing |

## Python API

| Function | Purpose |
| --- | --- |
| `eris.encode()` | Generate an explicit structural representation |
| `eris.embed()` | Generate a learned structural embedding |
| `eris.batch_encode()` | Encode a manifest of structures |
| `eris.batch_embed()` | Embed a manifest of structures |
| `eris.pack()` | Index generated artifacts |
| `eris.to_torch()` | Convert an artifact for PyTorch |
| `eris.to_pyg()` | Convert a graph artifact for PyG |

Discovery functions are `eris.list_representations()`, `eris.get_representation_info(name)`,
`eris.list_embedders()`, and `eris.get_embedder_info(name)`.

## Artifacts

Eris stores each generated representation as:

```text
artifact/
├── meta.json
└── data.npz
```

- `data.npz` contains the numerical representation.
- `meta.json` records the input, representation, parameters, outputs, shapes, features, and provenance.

Artifacts use Eris StructureArtifact Schema v2.

## Batch and downstream use

`eris.batch_encode()` and `eris.batch_embed()` process CSV manifests and record per-row success
or failure. `eris.pack()` creates an index that can be loaded with `eris.datasets.ErisFolderDataset`.

Use `eris.to_torch()` or `eris.to_pyg()` for downstream export. PyG export is available for graph artifacts; Eris does not infer graph topology from other representation types.

## Learn more

- [`docs/artifact_schema_v2.md`](docs/artifact_schema_v2.md): Artifact Schema v2 contract.
- [`docs/gearnet_edge_backend.md`](docs/gearnet_edge_backend.md): GearNet-Edge runtime and checkpoint contract.
- [`DEVELOPMENT.md`](DEVELOPMENT.md): local development information.

## License

Eris is released under the [MIT License](LICENSE).

## Acknowledgements

GearNet-Edge uses the official Multiview Contrast checkpoint `mc_gearnet_edge.pth` from
[Zenodo record 10.5281/zenodo.7723075](https://doi.org/10.5281/zenodo.7723075) and follows the
[GearNet](https://github.com/DeepGraphLearning/GearNet) model reference.
